"""This file contains the required code to load model's weigths"""
from .gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from .load_weigths import LoadWeigths

class LoadModelWeigths:
    def __init__(self):
     self.load_weigths = LoadWeigths()
     self.cfg =  {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1024,
        "n_layers":24,
        "n_heads":16,
        "drop_rate": 0.0, # Dropout rate
        "qkv_bias": True # Query-key-value bias
        }

    #config model and load weigths function
    def config_model_and_load_weigths(self):
        #based configuration
        

        CHOOSE_MODEL = "gpt2-medium (355M)"

        #remove the round brackets
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")") 

        #download and load the gpt2 model
        settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
        )

        model = GPTModel(self.cfg)
        self.load_weigths.load_weights_into_gpt(model, params)

        return model
