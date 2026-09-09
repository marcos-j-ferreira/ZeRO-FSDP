"""
interferência no modelo 
As informações são lidas do .yaml

"""

from main import Config

import torch
import numpy as np
import yaml
from typing import Dict , Any
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open("r", encoding="utf8") as file:
        return yaml.safe_load(file)

# Antes
#from model.transformers import ModeloCompleto
# novo
from main import ModeloCompleto


# Configurações do modelo
VOCAB_SIZE =  10001
EMBEDDING_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 4
BLOCK_SIZE = 260
DROPOUT = 0.0


from artifacts.Tokenizador import BPEPipeline

def main():

    
    config = Config(
        {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
            "dropout": DROPOUT,
        }
    )


    pipe = BPEPipeline().load_vocab("./artifacts")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModeloCompleto(config).to(device)


    model_save_path = "model_final.pth"

    if Path(model_save_path).exists():

        print(f"Carregando modelo de {model_save_path}")

        checkpoint = torch.load(
            model_save_path,
            map_location=device  
        )

        # Caso o checkpoint seja diretamente o state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    else:
        print(
            f"Arquivo de modelo não encontrado: "
            f"{model_save_path}. Treinando do zero."
        )


    prompts = "Lily was a young girl who loved to explore the"
    # output
    # Generated text: texto de entrada, como já relatado no centenário do

    print()

    for _ in range(3):
        model.eval()
        with torch.no_grad():
            input_ids = pipe.encode(prompts)
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)



            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated_text = pipe.decode(output[0].tolist())
            print("Generated text:", generated_text)
            print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
