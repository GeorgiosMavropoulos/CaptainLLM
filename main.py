##import tokenizer



from gptmodel.gpt_model import GPTModel,LayerNormalization 
from gptmodel.config import GPT_CONFIG_124M as cfg
import torch

##test DataLoad from pytorch
def main():

 """
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
  """

 ##test the gptmodel class
 #use the tokenizer to tokenize 2 texts
 import tiktoken as tk
 batch = []
 txt1 = "Every effort moves you"
 txt2 = "Every day holds a"
 
 ##apply the encoding
 tokenizer = tk.get_encoding("gpt2")
 batch.append(torch.tensor(tokenizer.encode(txt1)))
 batch.append(torch.tensor(tokenizer.encode(txt2)))
 batch = torch.stack(batch, dim=0) #concatenate the tensors into a bigger one with dim 0
 #print(batch)
 #create the embeddings and the context vector using the gpt model
 torch.manual_seed(123)
 model = GPTModel(cfg)
 logits = model(batch)
 #print("Output shape:", logits.shape)
 #print(logits)

 import torch.nn as nn
 torch.manual_seed(123)
 batch_example = torch.randn(2, 5)
 layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
 out = layer(batch_example)
 print(out)
 torch.set_printoptions(sci_mode=False,precision=20)
 

 ln = LayerNormalization(emb_dim=5)
 out_ln = ln(batch_example)
 mean = out_ln.mean(dim=-1, keepdim=True)
 var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
 print("Mean:\n", mean)
 print("Variance:\n", var)

main()




