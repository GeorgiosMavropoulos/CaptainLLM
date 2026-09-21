##import tokenizer
from tokenizer_bpe.__tokenizer import tokenizer
from dataloader.windowslider import LoadData
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
 initial_text = "Your journey starts with one step"
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
"""
 #calculate the scores
 sample = input_embeddings[0]
 query =  sample[1] # the second token input serves as a query
 attn_scores_2 = torch.empty(sample.shape[0]) ###empty the inputs in order to get filled by the loop
 for i, x_i in enumerate(sample):
   attn_scores_2[i] = torch.dot(x_i, query)##calculate the dot product and add it into the array
 #print(attn_scores_2) ## print computed attention scores

   #normalize using softmax
 attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
 #print("Attention weights:", attn_weights_2)
 #print("Sum:", attn_weights_2.sum())

 #calculate the context vector
 query = sample[1]
 context_vec_2 = torch.zeros(query.shape)
 for i,x_i in enumerate(sample):
   context_vec_2 += attn_weights_2[i]*x_i
   #print(context_vec_2)
"""

"""book example"""
inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)
   )

"""
##compute the scores
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
   attn_scores_2[i] = torch.dot(x_i, query)
#print(attn_scores_2) ##print scores

#normalize the scores 
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
#print("Attention weights:", attn_weights_2)
#print("Sum:", attn_weights_2.sum())

###calculate the context vector for the 'Journey' query
query = inputs[1]#the second input token is the query
context_vec_2 = torch.zeros(query.shape) ##create an empty vector
for i,x_i in enumerate(inputs):
   context_vec_2 += attn_weights_2[i]*x_i
   print(context_vec_2)
"""
"""
##compute scores, the weigths and the context vector for all the given embeddings
#use matrix multiplication to complete a fast computation

attn_scores = inputs @ inputs.T
#normalize using the softmax algorithm
attn_weights = torch.softmax(attn_scores, dim=-1)#normalize along the last dimension


##now multiply the embeddings with the attention weigths to compute context vector
context_vectors = attn_weights @ inputs
print(context_vectors)
"""
"""
##compute the attention mechanism with trainable weigths
x_2 = inputs[1] #journey's embedding
d_in = inputs.shape[1] #define the dimension input
d_out = 2# define the dimension out
 
#initialize 3 weight matrices Wq, Wk, Wv (weight query, weight key, weight value)
torch.manual_seed(123)
#create random parepemeters for each W
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

##compute the query, k and value vectors
query = x_2 @ W_query
key = x_2 @ W_key
value = x_2 @ W_value

#print(query)
##calculate key and values of all the inputs in order to be able to create the context vector of q2
keys = inputs @ W_key
values = inputs @ W_value

#print("keys.shape:", keys.shape)
#print("values.shape:", values.shape)

#compute the attention scores for the word Journey
keys_2 = keys[1]
attn_score_22 = query.dot(key) ##find the dot product between query matrice and key
#print(attn_score_22)

##generilize the attention score
attn_scores_2 = query @ keys.T
#print(attn_scores_2)

##calculate the attention weigths
d_k = keys.shape[-1] ##calculate along the last key
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) ##implement the softmax algorithm and divide the generilized attention score by the key's columns and rows in the power of 0.5
#print(attn_weights_2)

#compute context vectors by multiplying attention weigths with the random created values
context_vector = attn_weights_2 @ values
print(context_vector)
"""
#call the main method
main()
  ##test the embedding vectors
#input_ids = torch.tensor([2, 3, 5, 1])
#vocab_size = 50257
#output_dim = 256


 

