"""This class contains a method to evaluate the model on test questions"""
import torch
from gptmodel.gpt_model import GPTModel
from evaluate_model_script.config import GPT_CONFIG_124M as cfg
import json
from tokenizer_bpe.__tokenizer import _Tokenizer
from prompt_template.prompt_template import format_input
from training_scripts.generation import TextGeneration
import os
from dotenv import load_dotenv

load_dotenv()
generate_text = TextGeneration()

model = GPTModel(cfg)
class EvaluateModel:
    def __init__(self):
        self.model = model #initialize the fine tuned model
        self.device = self.detect_device() #detect user's device to run the model
        #transfer the model to device
        self.model.to(self.device)
        self.tokenizer = _Tokenizer()
        self.test_dataset = self.split_dataset() #split the given dataset
        self.load_checkpoint()
        print(next(self.model.parameters()).device)
        self.checkpoint_path = os.getenv("CHECKPOINT_PATH")
        

    #function to load the best checkpoint path
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval() #use eval() to disable training behaviors
        return self.model

    #method to split the dataset to evaluate
    def split_dataset(self):
        #load the dataset
        with open('C:/Users/Overkill/Desktop/train-llm/BaseLLM/InstructionLLM/alpaca_data.json') as file:
            #load the json
            full_dataset = json.load(file)

        #split the dataset
        test_dataset= full_dataset[:150]
        return test_dataset

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
    

   
    #evaluate model method
    
    def evaluate_model(self):
       #load the test dataset
       dataset = self.test_dataset

       #emtpy list to save all model's responses
       responses = []
       #iterate through the dataset
       for i in dataset:
          #format the dataset
          entry = format_input(i)
          
          print(entry)
          
          #tokenize the dataset
          encode_text = self.tokenizer.encoder(entry)
          #use torch tensor to create batch from the encoded entry
          encoded_entry = torch.tensor(encode_text).unsqueeze(0).to(self.device)
          print(encoded_entry.device)
          #use the generate text function to generate the output text
          returned_text = generate_text.generate_text(self.model,encoded_entry,max_new_tokens=250,context_size=cfg['context_length'],temperature=0.0,top_k=None, eos_id=50256)
          #convert from tensor to list the returned text
          tensor_to_list_text = returned_text.tolist()
          #convert the returned_text to string
          generated_text = self.tokenizer.decoder(tensor_to_list_text[0])
          #remove the prompt from the returned_response and keep only model's response
          returned_text_without_prompt = (
                        generated_text[len(entry):]
                        .replace("### Response:", "")
                        .strip()
                        )
          i["model_response"] = returned_text_without_prompt #keep model's response

          #save all data into a dictionary
          data = {
                          "instruction": i['instruction'],
                          "input": i['input'],
                          "expected_output": i['output'],
                          "model_response": i['model_response']
                      }
          responses.append(data)

       self.save_model_responses_from_test(responses) #save model's response


    
    #method to save model's responses and questions into a json file
    def save_model_responses_from_test(self,data):
       try:
        with open("instruction-data-with-response.json", "w", encoding="utf-8") as file:
               
                json.dump(data, file, indent=4,ensure_ascii=False)
                print("Data saved with success")
       except Exception as e:
           print(f"Error,can't save the data:{e}")


model_eval = EvaluateModel()
model_eval.evaluate_model()