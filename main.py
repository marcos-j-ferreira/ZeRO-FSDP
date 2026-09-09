"""
Arquitetura base - GPT-2

"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
    
class LayerNorm(nn.Module):
    """LayerNorm com suporte a bias opcional."""
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.ndim = ndim  # Guarda a dimensão
        #self.eps = eps  # Guarda o valor de epsilon
    def forward(self, input):
        # CORREÇÃO: usar a dimensão correta (a última dimensão)
        return F.layer_norm(input, (self.ndim,), self.weight, self.bias, 1e-5)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHead(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiHead(nn.Module):
    """Implementação MultiHead Attention para teste."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ModeloCompleto(nn.Module):
    """Versão simplificada do modelo para teste."""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Inicialização
        self.apply(self._init_weights)
        
        # Inicialização especial para projeções residuais
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Configuração do modelo
class Config:
    def __init__(self, config_dict):
        self.vocab_size = config_dict.get("vocab_size", 10000)
        self.n_embd = config_dict.get("embedding_dim", 128)
        self.num_head = config_dict.get("num_heads", 2)
        self.num_layer = config_dict.get("num_layers", 2)
        self.dropout = config_dict.get("dropout", 0.0)
        self.bias = config_dict.get("bias", False)
        self.block_size = config_dict.get("block_size", config_dict.get("seq_len", 128))
        self.num_experts = config_dict.get("num_experts", 8)
        self.num_experts_per_tok = config_dict.get("num_experts_per_tok", 4)
        self.moe_aux_loss_coef = config_dict.get("moe_aux_loss_coef", 0.01)

# Save final
def save_model(model_save_path, model):
    """Salva o modelo em FP16"""
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    fp16_state_dict = {}
    for key, value in state_dict.items():
        if value.is_floating_point():
            fp16_state_dict[key] = value.half()
        else:
            fp16_state_dict[key] = value
    
    torch.save(fp16_state_dict, model_save_path)
    print(f"Modelo salvo em FP16: {model_save_path}")

# Learning rate dinamico - boa pratica
def get_lr( step: int, lr: float, warmup_steps: int, max_steps: int, min_lr: float) -> float:
    """Warmup linear seguido de cosine decay até min_lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return min(lr * (step + 1) / warmup_steps, lr)

    if step >= max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Protege contra um erro de arredondamento de ponto flutuante na fronteira.
    return min(max(min_lr + coeff * (lr - min_lr), min_lr), lr)

@torch.no_grad()
def estimate_loss(model, loader, num_batches: int, device, amp_dtype, use_amp: bool) -> float:
    """Calcula a loss media em batches reservados para validacao."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = loader.get_batch()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)

# Save final
def save_model(model_save_path, model):
    """Salva o modelo em FP16"""
    path = Path(model_save_path)
    if str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    fp16_state_dict = {}
    for key, value in state_dict.items():
        if value.is_floating_point():
            fp16_state_dict[key] = value.half()
        else:
            fp16_state_dict[key] = value
    
    torch.save(fp16_state_dict, path)
    print(f"Modelo salvo em FP16: {path}")
