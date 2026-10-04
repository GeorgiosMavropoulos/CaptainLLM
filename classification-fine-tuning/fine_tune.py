###This is the main file where I load the model, load the pretrained weigths and fine tune the model
from gptmodel.config import GPT_CONFIG_124M as cfg
from download_datasets.gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from load_weigths.load_weigths import LoadWeigths
from train_script.generation import TextGeneration
from tokenizer_bpe.__tokenizer import _Tokenizer
import torch
class FineTune:
    def __init__(self):
      pass

    #initialize LoadWeigths class
    load_weigths = LoadWeigths()

    #initialize an object from TextGeneration class
    generate_text = TextGeneration()

    #create an instance of _Tokenizer class
    tokenizer = _Tokenizer()

    def config_model():

        model_configs = { ##available configurations for the GPT2 Model
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        gpt2_small_model = "gpt2-small (124M)"
        input = "Every effort moves"
        
        cfg.update(model_configs[gpt2_small_model]) #update configurations for the gpt2_small
        
        return gpt2_small_model, cfg

    #call config model method
    gpt2_small_model,cfg = config_model()
    
    model_size = gpt2_small_model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
    )
    model = GPTModel(cfg)
    load_weigths.load_weights_into_gpt(model, params)
    model.eval()

    ## validate that the model works and can generate coherent text
    text_2 = (
"Is the following text 'spam'? Answer with 'yes' or 'no':"
" 'You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award.'"
)
    encoded = tokenizer.encoder(text_2) #encode text
    token_ids = generate_text.generate_text(
    model=model,
    
    idx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0), #tensor the encoded text
    max_new_tokens=15,
    context_size=cfg["context_length"]
    )
    decoded_text = tokenizer.decoder(token_ids.squeeze(0).tolist())
    print(decoded_text)

    #let's validate the model on classification


    
    
    
   