"""This file is the main entry point to load the app in your terminal and run the llm"""
from gptmodel.gpt_model import GPTModel  #import the gpt2 model's architecture (attention mechanism,transforms blocks)
from gptmodel.config import GPT_CONFIG_124M  as cfg #import configuration file
import torch
from tokenizer_bpe.__tokenizer import _Tokenizer #import the tokinizer to tokenizer and decode texts
class Inference:
    def __init__(self):

      self.device = self.detect_device() #detect user's device

      #create an object from GPTModel class
      self.gpt_model = GPTModel(cfg)

      #transfer the model to user's device
      self.gpt_model.to(self.device)

      #update model's head ouput
      self.update_models_head_output()

      #create an object from tokenizer class
      self.tokenizer = _Tokenizer()

      #load the pretrained weigths
      self.load_weigths()

    #function to detect device
    def detect_device(self):
        # Auto-detect hardware (CUDA GPU, Apple Silicon MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        return self.device

    #this functions loads the .pth file which contains all the pretrained weigths for the classification task
    def load_weigths(self):
        checkpoint = torch.load(
        "checkpoint.pth",
        map_location=self.device
        )
        
        self.gpt_model.load_state_dict(
        checkpoint["model_state_dict"]
        )
        self.gpt_model.eval() #use eval() to disable training behaviors

    #update model's head output, since we want to output 2 tokens (o for ham and 1 for spam)
    def update_models_head_output(self):
        torch.manual_seed(123)
        num_classes = 2
        self.gpt_model.out_head = torch.nn.Linear(
        in_features=cfg["emb_dim"],
            out_features=num_classes
            )
        
        

    
inference = Inference()

print(inference.device)
print(inference.gpt_model.training)


