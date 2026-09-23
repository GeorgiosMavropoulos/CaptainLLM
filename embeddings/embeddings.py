### this file contains the embedding class which takes the original inputs, tokenize them, create the positional embeddings and the final input to add to the attention mechanism
import torch
from dataloader.windowslider import LoadData


class Embeddings:
    def __init__(self,initial_text,vocab_size,output_dim,context_length ):
        self.initial_text = initial_text
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_length = context_length
        #initialze the load data module
        self.load_data_pytorch = LoadData()


   

    #method to create the original token embeddings
    def create_token_embeddings(self): 
    
      token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.output_dim) #create the embedding layer
      ##call create_dataset_v1 from LoadData class 
      # in order to prepare the test and train the model using the auto-regressive method.
      #For this example I will use a small batch_size
      dataloader = self.load_data_pytorch.create_dataloader_v1(self.initial_text,
          batch_size=1,
          max_length=1024,
          stride=4,
          shuffle=False)
      
      #use the iter which uses python's next function 
      # in order to iterate through the batches and fetch the next entry
      data_iter = iter(dataloader)
      inputs, targets = next(data_iter)
      token_embeddings = token_embedding_layer(inputs) #create the original embeddings
      return token_embeddings
    
      
    
    def create_input_embeddings(self):
     
      
        #create the position embeddings that have the same embedding dimension as the token_embedding_layer
      pos_embedding_layer = torch.nn.Embedding(self.context_length, self.output_dim) ##create the embeddings and arrange each of embedding into one of the positions the algorithm below created
      pos_embeddings = pos_embedding_layer(torch.arange(self.context_length)) #create the number of weight's positions (4 in our case)
    
      #method to create token embeddings
      token_embeddings = self.create_token_embeddings()
    
      #add the original embeddings into the the pos_embeddings in each of the 8 batches
      input_embeddings = token_embeddings + pos_embeddings
      return input_embeddings



    