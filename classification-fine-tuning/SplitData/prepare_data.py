"""This file contains the required code to split data into training datasets, validation and testing"""
from dataloaders.data_loaders_pytorch import SpamDataSet
import tiktoken as tk
class PrepareTraining:
    def __init__(self):
        pass


    tokenizer = tk.get_encoding("gpt2") #get gpt2's encoding

    
    #create training dataset
    train_dataset = SpamDataSet(csv_file='C:/Users/Overkill/Desktop/train-llm/BaseLLM/classification-fine-tuning/datasets/train.csv', 
                                max_length=None,tokenizer=tokenizer)

    print(train_dataset.max_length)
