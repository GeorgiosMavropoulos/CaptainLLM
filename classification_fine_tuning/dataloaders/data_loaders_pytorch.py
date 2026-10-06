"""
Since we have to work with text messages of varying length we have to pad the text with less context. 
We are going to tokenize the text and we are going to find the batch with the highest length.
Then we are going to tokenize the special character "<|endoftext|>". The outcoming token will be added in the batches with smaller context length in order to
make the batches equal to the ones with the highset length

"""
import torch
from torch.utils.data import Dataset
import tiktoken as tk
import pandas as pd


class SpamDataSet(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,pad_token_id=50256):

     self.data = pd.read_csv(csv_file)
     self.tokenizer = tokenizer

     

     #pretokenize texts
     self.encoded_texts = [self.tokenizer.encode(text) for text in self.data["text"]]

     if max_length is None: #if max_length has not been delegated, set it equals to the longest encoded length
        self.max_length = self._longest_encoded_length()
     else: #if max length was delegated assign it to the delegated value
        self.max_length = max_length

        #truncate sequences if they are longer than max_length
        #multiply encoded_text + pad_token_id (50256) with max_length - the length of the encoded text 
     self.encoded_texts = [encoded_text + [pad_token_id] *  (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]


    def __getitem__(self, index):
        encoded = self.encoded_texts[index]  #get the indexed encoded text
        label = self.data.iloc[index]["label"] #access the label of the encoded text (0 for ham or 1 for spam)
        # Convert string labels to class indices
        if isinstance(label, str):
            label = label.strip().lower()

        if label == "ham":
            label = 0
        elif label == "spam":
            label = 1
        else:
            raise ValueError(f"Unknown label: {label}")
        
        return (
        torch.tensor(encoded, dtype=torch.long), #transform encoded text's tokens into tensors
        torch.tensor(label, dtype=torch.long) #transforer label's tokens into tensors
        )

    def __len__(self): #get corresponding data's length
            return len(self.data)   


    def _longest_encoded_length(self):  #this method retrieves longest text's length value to identify the max_length
        max_length = 0
        for encoded_text in self.encoded_texts:
         encoded_length = len(encoded_text)
         if encoded_length > max_length:
          max_length = encoded_length
        return max_length
                                