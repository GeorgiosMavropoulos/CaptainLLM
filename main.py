#import the train class

from train_script.train import Train
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
from train_script.prepare_data import PrepareData
from train_script.calculate_loss_functions import CalculateLoss
from tokenizer_bpe.__tokenizer import _Tokenizer
import torch

from gpt_download import download_and_load_gpt2


def main():


 if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

 #create an instance of GPTModel class
 model = GPTModel(cfg)

 #instnace of calculate loss class
 calculate_loss = CalculateLoss()
 #initialize the train class
 train = Train()

 #initialize the tokinizer
 tokenizer = _Tokenizer()


 #call the prepare dataset function from PrepareData class (prepare_data file) to split the dataset into trainable and validation data
  ##create an instance of prepare data class to split the text
 prepare_dataset = PrepareData()
 
 #filepath of the train text
 filepath = "C:/Users/Overkill/Desktop/train-llm/BaseLLM/datasets/dracula.txt"

 train_ratio = 0.90 
 #call prepare data method from prepare data class to split the data into trainable and validation data
 train_data,val_data = prepare_dataset.prepare_data_for_training(filepath,train_ratio)

 
 #if a cuda gpu is available train the llm on cuda, otherwise on the cpu
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 model.to(device) ##force model to train on the available device

 with torch.no_grad(): #disable gradient for efficiency since we are not training yet
  train_loss = calculate_loss.calc_loss_loader(train_data, model, device)
  val_loss = calculate_loss.calc_loss_loader(val_data, model, device)
 
 torch.manual_seed(123)
 """
 #call load weigths from gpt method
 settings, params = download_and_load_gpt2(
model_size="124M", models_dir="gpt2"
)
 """

 #start training function
 def start_training():
    #train for a third time
    #load the pretrained weigths
    checkpoint = torch.load("model_and_optimizer_updated.pth", map_location=device)
    #define a new model
    model3 = GPTModel(cfg)
    model3.load_state_dict(checkpoint["model_state_dict2"]) #load previous model's state
    optimizer3 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1) #apply AdamW's optimizer
    optimizer3.load_state_dict(checkpoint["optimizer_state_dict2"])

    num_epochs = 11
    train_losses, val_losses, tokens_seen = train.train_model_simple(
    model, train_data, val_data, optimizer3, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
    )
    model3.train()#train the model again
    torch.save({
    "model_state_dict3": model3.state_dict(),
    "optimizer_state_dict3": optimizer3.state_dict(),
    },
    "model_and_optimizer_updated2.pth"
    ) #save model's weigths and Adam's optimizers
 #call start training method
 #start_training()




 #list model's differences
 model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

 ##define the new model
 model_name = "gpt2-small (124M)"
 NEW_CONFIG = GPT_CONFIG_124M.copy()
 


 




main()




