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
 #implement the AdamW optimizer which penalizes larger weigths in order to avoid overfitting
 optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004, weight_decay=0.1)

 print("4. Optimizer created")

 num_epochs = 10
 train_losses, val_losses, tokens_seen = train.train_model_simple(
model, train_data, val_data, optimizer, device,
num_epochs=num_epochs, eval_freq=5, eval_iter=5,
start_context="Having had some time at my disposal when in", tokenizer=tokenizer
)
 
 print("Train batches:", len(train_data))
 print("Val batches:", len(val_data))
 print("Expected training steps:", len(train_data) * num_epochs)

 


 




main()




