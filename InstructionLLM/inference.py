"""Here is the main entry point where you can chat with the LLM"""
import torch
import os
from dotenv import load_dotenv
from gptmodel.gpt_model import GPTModel
from evaluate_model_script.config import GPT_CONFIG_124M as cfg
from tokenizer_bpe.__tokenizer import _Tokenizer
from prompt_template.prompt_template import format_input
from training_scripts.generation import TextGeneration
load_dotenv()
model = GPTModel(cfg)
generate_text = TextGeneration()
class Inference:
    def __init__(self):
     self.model = model #initialize the fine tuned model
     self.device = self.detect_device() #detect user's device to run the model
     #transfer the model to device
     self.model.to(self.device)
     self.tokenizer = _Tokenizer()
     self.checkpoint_path = os.getenv("CHECKPOINT_PATH")
     self.load_checkpoint()
    
     
   
     #function to load the best checkpoint path    
    def load_checkpoint(self):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval() #use eval() to disable training behaviors
            return self.model

    #method to detect device
    def detect_device(self):
           # Auto-detect hardware (CUDA GPU, Apple Silicon MPS, or CPU)
           if torch.cuda.is_available():
              self.device = torch.device("cuda")
           elif torch.backends.mps.is_available():
              self.device = torch.device("mps")
           else:
             self.device = torch.device("cpu")
           return self.device  


    #method to interact with the model
    def interact_with_model(self):
      #get user's input
      try:
          print("Model is active.Type exit to abandon the chat")
          instruction = ""
          while instruction != 'exit':
            
                        
            instruction = input().lower()
            #create a dictionary since input format expects a dictionary
            entry = {
                "instruction": instruction,
                "input": ""
            }
                        
                        #format the instruction
            formatted_entry = format_input(entry)
            #decode instruction so the model can use it
            encoded_instruction = self.tokenizer.encoder(formatted_entry )
            #create instruction into tensor and transfer the tensor to device
            encoded_instruction_to_tensor = torch.tensor(encoded_instruction).unsqueeze(0).to(self.device)

            #use the generate text function to generate the output text
            returned_text = generate_text.generate_text(self.model,
            encoded_instruction_to_tensor,max_new_tokens=250,context_size=cfg['context_length'],temperature=0.0,top_k=None, eos_id=50256)

            #convert from tensor to list the returned text
            tensor_to_list_text = returned_text.tolist()
            #convert the returned_text to string
            generated_text = self.tokenizer.decoder(tensor_to_list_text[0])

            #keep only model's response
            returned_text_without_prompt = (
                                    generated_text[len(formatted_entry ):]
                                    .replace("### Response:", "")
                                    .strip()
                                    )
            print(returned_text_without_prompt)
            
        
      except Exception as e:
            print(f"Error, cannot initiate the model:{e}")

#initialize the model's app
if __name__ == "__main__":
    ##call the inference to run the model
    inference = Inference()
    inference.interact_with_model()