# ZeRO e FSDP

ZeRO (Zero Redundancy Optimizer) e FSDP (Fully Sharded Data Parallel) são estratégias de paralelismo de dados que resolvem o principal gargalo do treinamento distribuído tradicional (DDP): a redundância de memória.

No DDP clássico, cada GPU mantém uma cópia completa de três coisas:
- **Pesos do modelo** (parâmetros)
- **Gradientes**
- **Estados do otimizador** (ex: momentos do Adam, que sozinhos já dobram o custo de memória dos parâmetros)

Isso significa que, com N GPUs, você replica N vezes algo que poderia ser dividido entre elas. ZeRO e FSDP atacam exatamente esse desperdício, particionando (*sharding*) esses três componentes entre os workers em vez de replicá-los.

O ZeRO (proposto pela Microsoft, no paper *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*) define isso em estágios progressivos:

- **ZeRO-1**: particiona apenas os **estados do otimizador**. Cada GPU cuida de uma fatia do otimizador para o modelo inteiro.
- **ZeRO-2**: além dos estados do otimizador, particiona também os **gradientes**.
- **ZeRO-3**: particiona também os **pesos do modelo**. Cada GPU só guarda uma fatia dos parâmetros o tempo todo, reconstruindo o restante sob demanda via comunicação.

O **FSDP** (implementação nativa do PyTorch) segue essencialmente a mesma ideia do ZeRO-3: os parâmetros ficam "shardados" (fragmentados) entre os GPUs e são reunidos (*gathered*) temporariamente apenas quando necessário para o forward/backward de uma camada, sendo descartados (*re-shardados*) logo em seguida.

Resumindo a progressão: **ZeRO-1 e ZeRO-2** economizam memória mantendo os pesos completos em cada GPU (menor troca de comunicação); **ZeRO-3/FSDP** economizam ainda mais memória, mas pagam o preço de mais comunicação, pois os pesos precisam ser reconstruídos a cada uso.


# Comunicação

A diferença central entre DDP e ZeRO/FSDP está em *quando* e *o que* é comunicado entre GPUs.

**DDP** faz basicamente uma operação por passo: **all-reduce** dos gradientes ao final do backward, somando (e depois dividindo pela quantidade de workers) os gradientes calculados em cada GPU para que todas terminem com o gradiente médio idêntico.

**ZeRO/FSDP** decompõe essa comunicação em operações mais granulares:

- **Reduce-scatter**: em vez de um all-reduce completo (que entrega o gradiente somado inteiro para todo mundo), o reduce-scatter soma os gradientes e distribui apenas a fatia correspondente a cada GPU. É mais barato em memória e ainda equivalente em volume de dados trafegados.
- **All-gather**: usado (principalmente no ZeRO-3/FSDP) para reconstruir temporariamente os pesos completos de uma camada antes do forward ou backward, já que cada GPU só guarda uma fatia deles.

Um ciclo típico de treino com FSDP (ZeRO-3), por camada, é:
1. **All-gather** dos pesos da camada (reconstrói o peso completo a partir dos shards).
2. Executa o forward com os pesos completos.
3. Descarta os pesos completos (mantém só o shard).
4. No backward, repete o all-gather, calcula os gradientes.
5. **Reduce-scatter** dos gradientes (cada GPU fica só com sua fatia do gradiente).
6. Atualiza apenas a fatia local dos pesos com a fatia local do otimizador.

Esse padrão de "all-gather antes de usar, descarta depois" é o que permite treinar modelos muito maiores que a memória de uma única GPU, ao custo de mais tráfego de rede — por isso a banda e a topologia de interconexão (NVLink, InfiniBand, etc.) importam tanto nesse regime.


# Dados

Do ponto de vista de dados, ZeRO/FSDP continuam sendo formas de **paralelismo de dados**: cada GPU processa um *shard* diferente do batch (um `DistributedSampler` ou equivalente), assim como no DDP.

A diferença é que, no DDP, cada GPU tem o modelo inteiro e processa seus próprios dados de forma totalmente independente até o momento do all-reduce dos gradientes. No ZeRO/FSDP, mesmo processando dados diferentes, as GPUs dependem umas das outras a cada camada para reconstruir os pesos (via all-gather), então o paralelismo de dados fica mais entrelaçado com a comunicação de parâmetros.

Isso não muda a lógica de particionamento dos dados em si (batch global dividido em micro-batches por GPU), mas aumenta a dependência de sincronização entre workers ao longo do forward/backward, não só ao final dele.


# DDP com ZeRO/FSDP

Vale reforçar: **ZeRO/FSDP não substituem o paralelismo de dados, eles o otimizam**. Tecnicamente, ZeRO-1/2/3 podem ser vistos como "DDP + sharding progressivo de memória".

Comparando:

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 / FSDP |
|---|---|---|---|---|
| Pesos replicados | Sim | Sim | Sim | Não (shardados) |
| Gradientes replicados | Sim | Sim | Não (shardados) | Não (shardados) |
| Estado do otimizador replicado | Sim | Não (shardado) | Não (shardado) | Não (shardado) |
| Comunicação extra | Baixa | Baixa | Média | Alta |
| Economia de memória | Nenhuma | Média | Boa | Máxima |

Na prática, isso também pode ser combinado com outras formas de paralelismo (tensor parallelism, pipeline parallelism) para modelos ainda maiores — o que costuma ser chamado de paralelismo 3D/4D —, mas ZeRO/FSDP sozinhos já resolvem boa parte do problema de memória para a maioria dos casos de treino em um único nó ou cluster pequeno.


# Mais eficiência

Algumas técnicas costumam ser combinadas com ZeRO/FSDP para ganhar ainda mais eficiência de memória e throughput:

- **Mixed precision (fp16/bf16)**: reduz pela metade a memória ocupada pelos pesos e ativações durante o forward/backward, mantendo uma cópia mestre em fp32 para a atualização do otimizador.
- **Activation checkpointing**: em vez de guardar todas as ativações intermediárias para o backward, recalcula parte delas sob demanda, trocando computação extra por memória.
- **CPU/NVMe offload**: parte dos estados do otimizador (ou até dos pesos) pode ser movida para a memória da CPU ou disco quando não está em uso imediato, ao custo de mais latência de transferência (ZeRO-Offload / ZeRO-Infinity).
- **Overlap de comunicação e computação**: sobrepor o all-gather da próxima camada com o cálculo da camada atual, escondendo parte da latência de rede.
- **Prefetching de parâmetros**: antecipar o all-gather dos pesos da próxima camada antes de precisar deles, reduzindo tempo ocioso de GPU esperando comunicação.

A escolha de qual estágio (ZeRO-1/2/3) e quais dessas técnicas usar depende diretamente do tamanho do modelo em relação à memória disponível por GPU e da banda de interconexão entre os nós.


# Inferência

Para inferência, normalmente **não faz sentido manter o sharding de ZeRO-3/FSDP**, já que:

- Não há gradientes nem estados de otimizador a manter — o principal ganho de memória do ZeRO desaparece.
- O all-gather constante de pesos a cada forward adicionaria latência desnecessária, prejudicando o tempo de resposta.

Por isso, o fluxo comum é: treinar com ZeRO-3/FSDP para caber o modelo na memória disponível, e depois **consolidar os shards em um checkpoint único** (pesos completos, sem particionamento) antes de servir o modelo para inferência. Para modelos que não cabem numa única GPU nem assim, aí sim entram estratégias específicas de inferência, como paralelismo de tensor ou pipeline, que são otimizadas para baixa latência — diferente do foco em throughput de treino do ZeRO/FSDP.
