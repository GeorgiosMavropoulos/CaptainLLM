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

      #update model's head ouput
      self.update_models_head_output()

      #transfer the model to user's device
      self.gpt_model.to(self.device) 

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

    #method to accept input and predict the classification
    def predict(self):
        #get input from the users
        sms = str(input("Insert the text please:"))
        #convert the text into token ids
        encoded_sms = self.tokenizer.encoder(sms)
        #truncate token's in encoded sms has more tokens than the model can handle
        truncated_sms = self.truncate_long_sequences(encoded_sms)
        #convert the sms into tensors
        input_ids = torch.tensor(truncated_sms).unsqueeze(0).to(self.device)

        #apply no gradients to predictions and get last values
        prediction = self.get_highest_predictions_disable_gradient(input_ids)

        #convert predicted tokens into spam or ham
        prediction_text = self.convert_token_to_text_prediction(prediction)
        
        print(prediction_text)
        return prediction_text
       

    #helper method to convert prediction into ham or spam
    def convert_token_to_text_prediction(self,prediction):
       # convert predicted label into spam or ham
                if prediction == 1:
                    return "spam"
                elif prediction == 0:
                   return 'ham'
                else:
                     raise ValueError(f"Invalid prediction,please try again later")


    
    #method to truncate long sequences so the input fits model's context size
    def truncate_long_sequences(self,input):
              #access model's context_length from config file
              supported_context_length = cfg['context_length']
               #truncate sequences if they are too long
              if len(input) > supported_context_length:
                input = input[:supported_context_length]  
                
              return input   

    #helper method to disable gradients and get the last values from the transformers block
    def get_highest_predictions_disable_gradient(self,input_ids):
         with torch.no_grad(): #do not apply any gradient
                 logits = self.gpt_model(input_ids)[:, -1, :] #get the last values from the transformers block
                 
         predicted_label = torch.argmax(logits, dim=-1).item()
         return predicted_label

                             
             
    
        
        

    
inference = Inference()
def menu():
     choice = ""
     while choice != "bye":
      print("Type text to provide a text to the LLM")
      print("Type bye to terminate the program")
      choice = str(input()).strip().lower()
      if choice == "text":
           inference.predict()
           
      elif choice == "bye":
           print("Goodbye")
           break
      else:
           print(f"Please choose between 'text' and 'bye'")

menu()



