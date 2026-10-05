#import the train class

from train_script.train import Train
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
from train_script.prepare_data import PrepareData
from train_script.calculate_loss_functions import CalculateLoss
from tokenizer_bpe.__tokenizer import _Tokenizer
import torch
from classification-fine-tuning.SplitData.prepare_data import PrepareTraining
from load_weigths.load_weigths import LoadWeigths
from download_datasets.gpt_download import download_and_load_gpt2

def main():

 ##create an instance of load weights class
 load_gpt_weigths  = LoadWeigths()

 #instnace of calculate loss class
 calculate_loss = CalculateLoss()
 #initialize the train class
 train = Train()

 #initialize the tokinizer
 tokenizer = _Tokenizer()

 

 #call the prepare dataset function from PrepareData class (prepare_data file) to split the dataset into trainable and validation data
  ##create an instance of prepare data class to split the text
 prepare_dataset = PrepareData()

  #list model's differences
 model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

 ##define the new model
 model_name = "gpt2-small (124M)"
 NEW_CONFIG = cfg.copy() # create a configuration file's copy
 #update config file
 NEW_CONFIG.update(model_configs[model_name])

 #update context_length to 1024 since this is the context length gpt uses
 NEW_CONFIG.update({"context_length": 1024})
 NEW_CONFIG.update({"qkv_bias":True})

 #instanciate a new gpt model
 gpt2_model= GPTModel(NEW_CONFIG)


 #load into our model gpt2's trained weigths
 def load_gpt2_weigths():
   #load gpt model
   settings, params = download_and_load_gpt2(
      model_size="124M", models_dir="gpt2"
      )
   load_gpt_weigths.load_weights_into_gpt(gpt2_model, params) #load the updated parameters into our gpt's instance
 

 #method to update gpt model and load the new weigths
 #start training function
 def start_training():

     load_gpt2_weigths()#call this method to update the model's weigths
     
    #filepath of the train text
     filepath = "C:/Users/Overkill/Desktop/train-llm/BaseLLM/datasets/the_verdict.txt"

     train_ratio = 0.90 
     #call prepare data method from prepare data class to split the data into trainable and validation data
     train_data,val_data = prepare_dataset.prepare_data_for_training(filepath,train_ratio)

     #if a cuda gpu is available train the llm on cuda, otherwise on the cpu
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     gpt2_model.to(device)

     with torch.no_grad(): #disable gradient for efficiency since we are not training yet
      train_loss = calculate_loss.calc_loss_loader(train_data, gpt2_model, device)
      val_loss = calculate_loss.calc_loss_loader(val_data, gpt2_model, device)
     
     torch.manual_seed(123)
     
     optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=5e-5, weight_decay=0.1)
     num_epochs = 5
     train_losses, val_losses, tokens_seen = train.train_model_simple(
     gpt2_model, train_data, val_data, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
    )
     gpt2_model.train()#train the model again

     torch.save({
        "model_state_dict": gpt2_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer_updated.pth"
        ) #save model's weigths and Adam's optimizers
    
     
 #call start training method
 start_training()


 
 


 




main()




