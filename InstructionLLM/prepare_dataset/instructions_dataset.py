"""This file contains a class which uses the format text from prompt templates and tokenizes all the batches """
import torch
from torch.utils.data import Dataset
from datasets.download_dataset_with_instructions import data
from prompt_template.prompt_template import format_input

class InstructionsDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []

        #use a for loop to iterate through data and change prompt templates using format text function
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            #tokenize the text
            encoded_text = self.tokenizer.encoder(full_text)
            #append text to the list
            self.encoded_texts.append(encoded_text)

    #return the encoded text
    def __getitem__(self, index):
        return self.encoded_texts[index]


    #return the length of the encoded texts
    def __len__(self):
        return len(self.data) 

        