"""This file is the application layer. Here I call the model from the load model module and I use the training script to train the model"""
import torch
from prepare_dataset.split_dataset import train_data, test_data,val_data
from text_generation.generation import TextGeneration
from prompt_template.prompt_template import format_input
from load_model.load_model import LoadModelWeigths
from tokenizer_bpe.__tokenizer import _Tokenizer
model_with_weigths = LoadModelWeigths()
class FineTune:
    def __init__(self):
        self.text_generation = TextGeneration
        #initialize the model
        self.model = model_with_weigths.config_model_and_load_weigths()
        self.cfg = model_with_weigths.cfg
        self.tokenizer = _Tokenizer()
        
   
    def generate_text_with_model(self):
        #test the already created model
        torch.manual_seed(123)
        input_text = format_input(val_data[0])
        encoded_text = self.tokenizer.encoder(input_text)
        #convert text to token ids
        token_ids = self.text_generation.generate_text(self.model,idx = torch.tensor(encoded_text).unsqueeze(0),max_new_tokens = 35,
                                                            context_size = self.cfg['context_length'],eos_id=50256,
                                                            )
        #convert model's response to text
        #first convert the returned tensor to a list
        output_tokens = token_ids[0].tolist()
        generated_resposne = self.tokenizer.decoder(output_tokens)
        
        response_text = generated_resposne[len(input_text):].strip()
        print(response_text)
        return response_text

    #cann generate_text method
fine_tune = FineTune()
fine_tune.generate_text_with_model()

   