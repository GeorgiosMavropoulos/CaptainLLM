### This file contains the pretrained tokenizer BPE. This tokinzer is being implemented through 
##import tiktoken
import tiktoken as tk
#create tokenizer class
class _Tokenizer:
    def __init__(self):
        self.tokenizer = tk.get_encoding("gpt2")
        self.integer_ids = []
        self.strings = []
        self.text = ""
        


    #method to check tokenizer
    def encoder(self,text:str)->list[int]:
        
        #tokenize the text
        integer_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return integer_ids

    #decode method
    def decoder(self,token_ids:list[int])->str:
     
     strings = self.tokenizer.decode(token_ids)
     return strings

# Create an instance
tokenizer = _Tokenizer()


