##import tokenizer

from dataloader.windowslider import LoadData
from multiheadattention import multiheadattention
from embeddings.embeddings import Embeddings

import torch


##test DataLoad from pytorch
def main():


 #instanciate LoadData 
 load_data_pytorch = LoadData()

 def load_text():
  with open("datasets/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
  return raw_text


 
 #load text
 initial_text = load_text()
 vocab_size = 50257
 output_dim = 256
 context_length = 1024


 #initialize embeddings class
 embeddings = Embeddings(initial_text,vocab_size,output_dim,context_length)

 #create the input embeddings
 input_embeddings = embeddings.create_input_embeddings()
  

 
 #apply the multihead attention mechanism to calculate the weigths
 def apply_multihead_attention_mechanism():
   ##call the method to calculate the positional embeddings
    
  batch = input_embeddings
  #d_in = 3
  d_out = 768
  
  
  torch.manual_seed(123)
  batch_size, context_length,  d_in = batch.shape

  #use multiheadattention class wraper and create two instances
  mha = multiheadattention.MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
  context_vecs = mha(batch)
  return context_vecs
  

  #call the apply_multihead_attention_mechanism() to create the context vector
 print(apply_multihead_attention_mechanism())



main()




