"""
Treinamento com múltiplas GPUs usando FullyShardedDataParallel (FSDP) — equivalente
ao ZeRO (DeepSpeed) nativo do PyTorch.

Rodar com:
    torchrun --nproc_per_node=2 train_fsdp.py

Principais mudanças em relação à versão DDP:
  1. FSDP no lugar de DDP: em vez de cada GPU guardar uma cópia COMPLETA do
     modelo + gradientes + estado do otimizador, cada GPU guarda só a SUA
     fatia (shard). O restante é buscado via all-gather sob demanda no
     forward/backward e descartado logo em seguida.
  2. sharding_strategy define o "nível de ZeRO":
       - ShardingStrategy.FULL_SHARD    -> equivalente a ZeRO-3 (shard de
         params, grads e optimizer states). Máxima economia de memória.
       - ShardingStrategy.SHARD_GRAD_OP -> equivalente a ZeRO-2 (shard de
         grads e optimizer states, params replicados). Menos comunicação,
         mais memória.
       - ShardingStrategy.NO_SHARD      -> equivalente ao DDP puro.
       - ShardingStrategy.HYBRID_SHARD  -> shard dentro do nó, replica entre
         nós (bom para multi-node).
  3. auto_wrap_policy: decide COMO o modelo é particionado em unidades FSDP.
     Sem isso, o modelo inteiro vira uma única unidade e você perde boa
     parte do benefício de memória. O ideal é envolver cada bloco do
     transformer individualmente — ajuste `transformer_auto_wrap_policy`
     abaixo para apontar para a classe real do seu bloco (ex: Block,
     TransformerBlock, DecoderLayer...). Deixei um fallback por tamanho
     (size_based_auto_wrap_policy) caso você não tenha essa classe à mão.
  4. Clipping de gradiente: NÃO use mais
     torch.nn.utils.clip_grad_norm_(model.parameters(), ...) — com params
     fatiados isso dá norma errada. Use o método próprio do FSDP:
     model.clip_grad_norm_(GRAD_CLIP).
  5. model.module não existe mais como "unwrap direto". Para obter o
     state_dict completo (ex: para salvar), é preciso usar o context
     manager FSDP.state_dict_type(...) com FullStateDictConfig, que faz um
     all-gather e materializa o state dict completo (por padrão só no
     rank 0, com offload pra CPU).
  6. mixed_precision: se quiser usar BF16/FP16, isso é configurado via
     MixedPrecision no próprio FSDP (em vez de autocast manual), pois o
     FSDP também precisa saber em que dtype fazer o all-gather.
  7. device_id=local_rank: substitui o .to(device) + device_ids do DDP —
     o FSDP já move os shards para a GPU certa internamente.
"""

# imports
from main import ModeloCompleto
from main import Config
from main import get_lr
from main import save_model

# imports library
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os
import time
import numpy as np

# Dados
DATASET = os.path.dirname(os.path.abspath(__file__))
FILENAME = "train.bin"

# Configurações do modelo
VOCAB_SIZE = 10001
EMBEDDING_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
BLOCK_SIZE = 360
DROPOUT = 0.0

# configurações do treinamento
MAX_STEPS = 500
BATCH_SIZE = 12          # batch LOCAL por GPU
GRAD_ACCUM_STEPS = 20
WEIGHT_DECAY = 0.0001
WARMUP_STEPS = 10
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
GRAD_CLIP = 1.0

# logs
PRINT_INTERVAL = 1
EVAL_INTERVAL = 100

# Configuração do hardware
USE_AMP = False
USE_BF16 = False           # se True, ativa MixedPrecision do FSDP em bf16
TORCH_COMPILE = False       # se for usar torch.compile, use_orig_params=True abaixo é obrigatório

# ZeRO stage / sharding strategy
# "FULL_SHARD" = ZeRO-3 | "SHARD_GRAD_OP" = ZeRO-2 | "NO_SHARD" = DDP
SHARDING_STRATEGY = ShardingStrategy.FULL_SHARD

# save model
STEP_SAVE_INTERVAL = 500
SAVE_DIR = "model_final.pth"


# ---------------------------------------------------------------------------
# setup do process group + leitura de rank/local_rank/world_size (igual DDP)
# ---------------------------------------------------------------------------
def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def ddp_cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# get_batch idêntico ao da versão DDP: cada rank sorteia só na sua fatia
# lógica do dataset.
# ---------------------------------------------------------------------------
data_dir = DATASET
def get_batch(split, batch_size, block_size, device, rank, world_size):
    filename = FILENAME if split == 'train' else 'val.bin'
    path = os.path.join(data_dir, filename)
    if split == 'val' and not os.path.exists(path):
        path = os.path.join(data_dir, FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")

    data = np.memmap(path, dtype=np.uint16, mode='r')
    total_len = len(data) - block_size
    if total_len <= 0:
        raise ValueError(f"{path} possui poucos tokens para BLOCK_SIZE={block_size}.")

    shard_len = total_len // world_size
    shard_start = rank * shard_len
    shard_end = shard_start + shard_len

    ix = torch.randint(shard_start, shard_end, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, device, rank, world_size, num_batches=10):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch("val", BATCH_SIZE, BLOCK_SIZE, device, rank, world_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Salvar checkpoint com FSDP: precisa "des-fatiar" (all-gather) os params
# antes de chamar state_dict(). offload_to_cpu + rank0_only evitam estourar
# a memória da GPU/host durante esse gather.
# ---------------------------------------------------------------------------
def save_fsdp_model(path, model, is_main_process):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state_dict = model.state_dict()

    if is_main_process:
        # Ajuste aqui conforme a assinatura real do seu save_model():
        # se ele espera um objeto nn.Module (com .state_dict()), construa
        # uma instância "crua" do modelo e dê load_state_dict antes de salvar.
        torch.save(full_state_dict, path)
        print(f"Checkpoint salvo em: {path}")


def main():
    rank, local_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    config = Config({
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "dropout": DROPOUT,
        "rope_theta": 10001,
    })

    # IMPORTANTE: com FSDP, NÃO faça model.to(device) antes de envolver com
    # FSDP quando usar device_id — o FSDP cuida do posicionamento na GPU
    # correta ao materializar cada shard. (Se seu modelo for grande demais
    # para caber inteiro numa GPU só ao ser instanciado, considere criar em
    # meta device — fora do escopo deste ajuste pontual.)
    model = ModeloCompleto(config)

    # -----------------------------------------------------------------
    # auto_wrap_policy: troque pelo bloco real do seu transformer se
    # tiver a classe disponível, ex:
    #
    #   from main import Block
    #   from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    #   auto_wrap_policy = functools.partial(
    #       transformer_auto_wrap_policy,
    #       transformer_layer_cls={Block},
    #   )
    #
    # Isso garante que cada bloco vira sua própria unidade FSDP (melhor
    # granularidade = melhor overlap de comunicação/computação e menos
    # pico de memória). O fallback abaixo funciona sem conhecer a classe,
    # mas é menos eficiente.
    # -----------------------------------------------------------------
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    mixed_precision_policy = None
    if USE_BF16:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=SHARDING_STRATEGY,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=TORCH_COMPILE,  # obrigatório se for usar torch.compile
    )

    if is_main_process:
        parametros = int(sum(p.numel() for p in model.parameters()))
        print(f"Modelo: FSDP ({SHARDING_STRATEGY.name}) | Parâmetros: {parametros:,}")
        print(f"World size: {world_size} | Batch local: {BATCH_SIZE} | "
              f"Batch efetivo global: {BATCH_SIZE * world_size}")
        print(f"{MAX_STEPS} steps | Grad Accum Steps: {GRAD_ACCUM_STEPS} | "
              f"LR: {LEARNING_RATE} | Min LR: {MIN_LR} | Warmup Steps: {WARMUP_STEPS}")
        print()

    # O otimizador SEMPRE deve ser criado DEPOIS de envolver o modelo com
    # FSDP, pois model.parameters() aqui já reflete os params fatiados.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(WEIGHT_DECAY),
    )

    model.train()
    step = 0
    val_loss = 0.0
    tokens_seen = 0

    tokens_per_step = BLOCK_SIZE * BATCH_SIZE * GRAD_ACCUM_STEPS * world_size

    while step < MAX_STEPS:
        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        step_loss_accum = 0.0

        lr = get_lr(step, LEARNING_RATE, WARMUP_STEPS, MAX_STEPS, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch("train", BATCH_SIZE, BLOCK_SIZE, device, rank, world_size)

            logits, loss = model(x, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss não finita no step {step}: {loss.item()}")

            step_loss_accum += loss.item()

            loss_scaled = loss / GRAD_ACCUM_STEPS
            loss_scaled.backward()
            # O reduce-scatter dos gradientes entre GPUs acontece aqui dentro
            # do .backward(), assim como o all-reduce acontecia no DDP — só
            # que agora cada GPU fica só com a fatia do gradiente que lhe
            # pertence, em vez da cópia completa.

        grad_norm = 0.0
        if GRAD_CLIP is not None:
            # MUDANÇA CRÍTICA: com FSDP, o clipping tem que ser feito pelo
            # método do próprio FSDP, que sabe como calcular a norma global
            # correta a partir dos gradientes fatiados. Usar
            # torch.nn.utils.clip_grad_norm_ aqui daria um resultado errado.
            grad_norm = model.clip_grad_norm_(GRAD_CLIP)

        optimizer.step()

        torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_seen += tokens_per_step
        tokens_per_sec = tokens_per_step / dt

        if is_main_process:
            if step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, device, rank, world_size)
                print(f"Validação | Step {step} | Val Loss: {val_loss:.4f} | LR: {lr:.10f}")

            if step % PRINT_INTERVAL == 0:
                print(
                    f"Step {step} | Loss: {step_loss_accum / GRAD_ACCUM_STEPS:.4f} | "
                    f"VAL Loss: {val_loss:.4f} | LR: {lr:.10f} | "
                    f"norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | "
                    f"tok/s: {tokens_per_sec:,.0f} | tokens vistos: {tokens_seen:,}"
                )

        step += 1

    if is_main_process:
        print("Treinamento finalizado.")

    # save_fsdp_model já lida com o all-gather + salvar só no rank 0
    save_fsdp_model(SAVE_DIR, model, is_main_process)

    ddp_cleanup()


if __name__ == "__main__":
    main()
