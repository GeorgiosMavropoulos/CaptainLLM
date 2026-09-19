##This file contains the class which implements the dataloader and data set
##The general scope of this class is to create the window slider
from  tokenizer_bpe.__tokenizer import tokenizer

##import PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

##the GPTDatasetV1 is responsible to chunk the training text 
# into multiple pieces and assign the lists into tensors
class GPTDatasetV1(Dataset):
    ##initialize the constructor
    def __init__(self,txt,__tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        #tokenize the entire text
        token_ids = tokenizer.encoder(txt)

        #sliding window chunk
        def chunk_book():
            #use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride): ##loop through the length of token ids - max_length
                input_chunk = token_ids[i:i + max_length] 
                target_chunk = token_ids[i + 1: i + max_length + 1]
                ##
                self.input_ids.append(torch.tensor(input_chunk)) ##create the input chunk
                self.target_ids.append(torch.tensor(target_chunk)) ##create the target chunk
        #execute chunk book
        chunk_book()

        #return the total number of rows in the dataset
    def __len__(self): 
         return len(self.input_ids)

        #return a single row from the dataset
    def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]


   
##dataloader
class LoadData():
     def __init__(self):
      pass


     ###create the loader. assign the txt as an argument, batch size for to create 4 batches of training data,max length 256, shuffle to mix tokes
        ### stride = 128 means that each batch will contain 128 tokens since 4 x 128 = context of the dataset. 
        ## Drop last drops the last incomplete batch. If there isn't any incomplete batch it does not drop batches
     def create_dataloader_v1(self,txt, batch_size=4, max_length=256,
            stride=128, shuffle=True, drop_last=True,
            num_workers=0):    

         #create a tokenizer instance
         __tokenizer = tokenizer
         ##initialize GPTDatasetV1
         dataset = GPTDatasetV1(txt, __tokenizer, max_length, stride)  
         ##load the data
         dataloader = DataLoader(
             dataset,##input the loaded dataset
             batch_size=batch_size, #input the predefined batch size
             shuffle=shuffle,#initialize an instance of shuffle mode
             drop_last=drop_last,
             num_workers=num_workers)

         #return the dataloader
         return dataloader

   




  











"""
    ##let's try to test it first

    #load the data from the verdict story
    with open("datasets/the_verdict.txt", "r", encoding="utf-8") as f:
      raw_text = f.read()

    #encode text using the tokenizer
    encoded_text = tokenizer.encoder(raw_text)

    #remove the 50 first tokens for simplicity
    enc_sample = encoded_text[50:]

    ##create 2 variables X and Y. X contains the input and Y the target values
    context_size = 4 ##determine how many tokens each input contains
    #define the X input tokens
    X = enc_sample[:context_size] ##split the sample into lists with 4 tokens each
    Y = enc_sample[1:context_size + 1] #start from the first index and stop into the 5th (context size = 4 + 1)
        #print the context and the target as a text
    for i in range(1,context_size + 1):
           context = enc_sample[:i] ##take each sample
           desired = enc_sample[i] #take the target
           print(tokenizer.decoder(context), "---->", tokenizer.decoder([desired])) ## print the input and the its target




    print(X)
    print(Y)
    
    print('---------------')
    #create the next token tasks
    for i in range(1,context_size + 1):
       context = enc_sample[:i] ##take each sample
       desired = enc_sample[i] #take the target
       print(context, "---->", desired) ## print the input and the its target
       "
"""

    





   