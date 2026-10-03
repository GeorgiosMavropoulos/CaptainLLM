#import the train class

from train_script.train import Train
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
from train_script.prepare_data import PrepareData
from train_script.calculate_loss_functions import CalculateLoss
from tokenizer_bpe.__tokenizer import _Tokenizer
import torch




def main():

 print("PyTorch version:", torch.__version__)
 print("CUDA available:", torch.cuda.is_available())
 print("CUDA version:", torch.version.cuda)

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
 print("1. Starting main")

 #call the prepare dataset function from PrepareData class (prepare_data file) to split the dataset into trainable and validation data
  ##create an instance of prepare data class to split the text
 prepare_dataset = PrepareData()
 
 #filepath of the train text
 filepath = "C:/Users/Overkill/Desktop/train-llm/BaseLLM/datasets/dracula.txt"

 train_ratio = 0.90 
 #call prepare data method from prepare data class to split the data into trainable and validation data
 train_data,val_data = prepare_dataset.prepare_data_for_training(filepath,train_ratio)

 print("2. Data loaded")
 print("Train batches:", len(train_data))
 print("Val batches:", len(val_data))
 
 #if a cuda gpu is available train the llm on cuda, otherwise on the cpu
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("Device:", device)
 model.to(device) ##force model to train on the available device

 with torch.no_grad(): #disable gradient for efficiency since we are not training yet
  train_loss = calculate_loss.calc_loss_loader(train_data, model, device)
  val_loss = calculate_loss.calc_loss_loader(val_data, model, device)
 
 torch.manual_seed(123)
 """
 #implement the AdamW optimizer which penalizes larger weigths in order to avoid overfitting
 optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004, weight_decay=0.1)

 print("4. Optimizer created")

 num_epochs = 10
 train_losses, val_losses, tokens_seen = train.train_model_simple(
model, train_data, val_data, optimizer, device,
num_epochs=num_epochs, eval_freq=5, eval_iter=5,
start_context="Having had some time at my disposal when in", tokenizer=tokenizer
)
<<<<<<< HEAD
=======
<<<<<<< HEAD
 torch.save({
"model_state_dict": model.state_dict(),
"optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth"
) #save model's weigths and Adam's optimizers


 #load the pretrained weigths
 checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
 #define a new model
 model2 = GPTModel(cfg)
 model2.load_state_dict(checkpoint["model_state_dict"]) #load previous model's state
 optimizer2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1) #apply AdamW's optimizer
 optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

 num_epochs = 10
 train_losses, val_losses, tokens_seen = train.train_model_simple(
 model, train_data, val_data, optimizer2, device,
 num_epochs=num_epochs, eval_freq=5, eval_iter=5,
 start_context="Every effort moves you", tokenizer=tokenizer
 )
 model2.train()#train the model again
 torch.save({
 "model_state_dict2": model.state_dict(),
 "optimizer_state_dict2": optimizer2.state_dict(),
 },
 "model_and_optimizer_updated.pth"
 ) #save model's weigths and Adam's optimizers
 """
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

 


 




main()




