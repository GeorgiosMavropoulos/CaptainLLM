##import tokenizer
from tokenizer_bpe.__tokenizer import tokenizer
from dataloader.windowslider import LoadData
from self_attention  import attention_mechanism
#main class
"""
def main():

   #text example
   text = (
           "Akwirw ier"
           )
   #call tokenizer method
   text_to_id = tokenizer.encoder(text)

   ##print all id's
   for i in text_to_id:
      print(i)
   for token in text_to_id:
      print(tokenizer.decoder([token]))

   #call the decoder method
   id_to_string = tokenizer.decoder(text_to_id)

 

   #print results
   #print(text_to_id)
   print(id_to_string)

#execute main
main()
   """
import torch
import torch.nn as nn
torch.manual_seed(123)
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
 token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #create the embedding layer
 ##call create_dataset_v1 from LoadData class 
 # in order to prepare the test and train the model using the auto-regressive method.
 #For this example I will use a small batch_size
 dataloader =load_data_pytorch.create_dataloader_v1(initial_text,
    batch_size=1,
    max_length=4,
    stride=4,
    shuffle=False)
 
 #use the iter which uses python's next function 
 # in order to iterate through the batches and fetch the next entry
 data_iter = iter(dataloader)
 inputs, targets = next(data_iter)
 token_embeddings = token_embedding_layer(inputs) #create the original embeddings
 context_length = 4 # give the value of the max length
 #print(token_embeddings.shape)

  #create the position embeddings that have the same embedding dimension as the token_embedding_layer
 pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) ##create the embeddings and arrange each of embedding into one of the positions the algorithm below created
 pos_embeddings = pos_embedding_layer(torch.arange(context_length)) #create the number of weight's positions (4 in our case)
 
 #add the original embeddings into the the pos_embeddings in each of the 8 batches
 input_embeddings = token_embeddings + pos_embeddings

 inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your (x^1)
      [0.55, 0.87, 0.66], # journey (x^2)
      [0.57, 0.85, 0.64], # starts (x^3)
      [0.22, 0.58, 0.33], # with (x^4)
      [0.77, 0.25, 0.10], # one (x^5)
      [0.05, 0.80, 0.55]] # step (x^6)
      )

 d_in = 3
 d_out = 2
 batch = torch.stack((inputs, inputs), dim=0)
 
 torch.manual_seed(123)
 context_length = batch.shape[1]
 ca_v2 = attention_mechanism.CasualAttentionV1(d_in, d_out,context_length,0.0)
 context_vecs = ca_v2 (batch)
 print(context_vecs.shape)


 


main()
  ##test the embedding vectors
#input_ids = torch.tensor([2, 3, 5, 1])
#vocab_size = 50257
#output_dim = 256




