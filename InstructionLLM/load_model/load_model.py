"""This file contains the required code to load model's weigths"""
from .gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from .load_weigths import LoadWeigths
load_weigths = LoadWeigths()
#based configuration
cfg = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1024,
"n_layers":24,
"n_heads":16,
"drop_rate": 0.0, # Dropout rate
"qkv_bias": True # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-medium (355M)"

#create a dictionary with all models and their configurations
model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

#remove the round brackets
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")") 

#download and load the gpt2 model
settings, params = download_and_load_gpt2(
model_size=model_size,
models_dir="gpt2"
)

model = GPTModel(cfg)
load_weigths.load_weights_into_gpt(model, params)
print(model.tok_emb.weight.shape)