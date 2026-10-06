"""This file contains the required code to split data into training datasets, validation and testing"""
from ..dataloaders.data_loaders_pytorch import SpamDataSet
import tiktoken as tk
class PrepareTraining:
    def __init__(self):
        pass


    tokenizer = tk.get_encoding("gpt2") #get gpt2's encoding

    
    #create training dataset
    train_dataset = SpamDataSet(csv_file='C:/Users/Overkill/Desktop/train-llm/BaseLLM/classification_fine_tuning/datasets/train_new.csv', 
                                max_length=None,tokenizer=tokenizer)

    print(train_dataset.max_length)

    #create the validation dataset 
    val_dataset = SpamDataSet(
        csv_file="C:/Users/Overkill/Desktop/train-llm/BaseLLM/classification_fine_tuning/datasets/val_new.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
      )

    print(val_dataset.max_length)

    

   #dataset for tests
    test_dataset = SpamDataSet(
        csv_file="C:/Users/Overkill/Desktop/train-llm/BaseLLM/classification_fine_tuning/datasets/test_new.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
        )

    print(test_dataset.max_length)

   
