"""This file is the application layer. Here I call the model from the load model module and I use the training script to train the model"""
import torch
from dataloaders.dataloaders import torch_loaders
from tokenizer_bpe.__tokenizer import _Tokenizer

from training_scripts.calculate_loss_functions import CalculateLoss
from prompt_template.prompt_template import format_input
from load_model.load_model import LoadModelWeigths
from training_scripts.train import Train
from prepare_dataset.split_dataset import val_data
from time import time
model_with_weigths = LoadModelWeigths()


class FineTune:
    def __init__(self):
        
        #initialize the model
        self.model = model_with_weigths.config_model_and_load_weigths()
        self.cfg = model_with_weigths.cfg
        self.tokenizer = _Tokenizer()
        self.calculate_loss = CalculateLoss()
        #get loaders
        self.train_loaders,self.val_loader,self.test_loader = torch_loaders.get_loaders()
        self.device = torch_loaders.device
        self.trainer = Train()
   

        
    #train loop
    def start_training(self):
     start_time = time()
     #access model's device
     torch.manual_seed(123)
     #define the num of epochs
     self.model.to(torch_loaders.device)

     #define the optimizer
     optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
     num_epochs = 2
     #use train script to start training
     self.trainer.train_model(self.model,
    self.train_loaders,self.val_loader,optimizer,self.device,num_epochs,eval_freq=5, eval_iter=5,start_context=format_input(val_data[0]),tokenizer=self.tokenizer)
     end_time = time()
     execution_time_minutes = (end_time - start_time) / 60
     print(f"Training completed in {execution_time_minutes:.2f} minutes.")


   
   
  


fine_tune = FineTune()
fine_tune.start_training()


   