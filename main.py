
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
import torch
from gptmodel.gpt_model import  TransformerBlock
import tiktoken as tk
##test DataLoad from pytorch
def main():


 ##test the gpt model
 torch.manual_seed(123)
 batch = torch.randint(
    0,
    cfg["vocab_size"],
    (2, 4)
)
 model = GPTModel(cfg) #instanciate a model using the gpt model with gpt_config_124m


 #out = model(batch)
 
 #predict the next token to generate text
 #idx is a batch (batch,n_tokens) array of indices in the current context
 def generate_text_simple(model,idx,max_new_tokens,context_size):
  for _ in range(max_new_tokens):
   idx_cond = idx[:, -context_size:] ##abstract the context_size from the idx, in order not to outnumber the given context_length
   with torch.no_grad(): ##do not apply gradient
    logits = model(idx_cond) #create the logits

   logits = logits[:, -1, :] #focuses only on the last time step, so that (batch, n_tokens, vocabulary_size) becomes (batch, vocabulary_size)
   #convert the tokens to a propability distribution using softmax
   propabilities = torch.softmax(logits, dim=-1)
   #calculate the next token by finding the highest propability token using torch.argmax
   idx_next = torch.argmax(propabilities, dim=-1, keepdim=True)
   idx = torch.cat((idx, idx_next), dim=1) #append the sampled index to the running sequence, where idx has shape (batch, n_tokens +1)

  return idx #idx has shape (batch,1)

 ##training loss calculation
 def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

 def token_ids_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0)
  return tokenizer.decode(flat.tolist())

 start_context = "Every effort moves you"
 tokenizer = tk.get_encoding("gpt2")

 token_ids = generate_text_simple(
 model=model,
 idx=text_to_token_ids(start_context, tokenizer),
 max_new_tokens=10,
 context_size=cfg["context_length"]
)
 #print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#create the inputs into torch tensor
 #inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
#[40, 1107, 588]]) #"I really like"]

 #targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
#[1107, 588, 11311]]) # " really like chocolate"]

 #with torch.no_grad():#do not apply gradient since we are not training yet 
  #logits = model(inputs) #create the logits
 #probas = torch.softmax(logits, dim=-1) #propability of each token in vocabulary
 #print(probas.shape)

# token_ids = torch.argmax(probas, dim=-1, keepdim=True) #calculate the token ids with the highest propability score

 #print propability scores for each text
 #text_idx = 0
 #target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
 #print("Text 1:", target_probas_1)
 #text_idx = 1
 #target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
 #print("Text 2:", target_probas_2)

 #create the logarithm based on propabilites. we concatenate the two torches
 #log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
 #print(log_probas)

 #find the average of log_propabilities
 #avg_log_probas = torch.mean(log_probas)
 #print(avg_log_probas)

 #convert the negative number occured from the mean of probabilities into a positive number (cross entropy loss)
 #neg_ang_log_probas = avg_log_probas * -1
 #print(neg_ang_log_probas)

 #use cross entropy function to avoid all these steps
 #get logits and target token's shape
 #print("Logits shape:", logits.shape)
 #print("Targets shape:", targets.shape)

#mock some vectors
 inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
 [40, 1107, 588]]) #"I really like"]
 
 targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
 [1107, 588, 11311]]) # " really like chocolate"]

 with torch.no_grad():#do not apply gradient since we are not training yet 
   logits = model(inputs) #create the logits

 #flatten the tensors before apply the cross entropy function
 logits_flat = logits.flatten(0, 1)
 targets_flat = targets.flatten()
 #print("Flattened logits:", logits_flat.shape)
 #print("Flattened targets:", targets_flat.shape)

 #calculate the loss using cross entropy function
 #using cross entropy we save so many lines of code

 
 loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
 print(loss)
 perplexity = torch.exp(loss)
 print(perplexity)


 

main()




