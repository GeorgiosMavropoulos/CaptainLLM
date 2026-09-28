
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

 #delegate into a variable the output context vector
 out = model(batch)
 #print("Input batch:\n", batch)
 #print("\nOutput shape:", out.shape)
 #print(out)

 #get model's total paremeters
 total_params = sum(p.numel() for p in model.parameters())
 #print(f"Total number of parameters: {total_params:,}")

 total_params_gpt2 = (
  ##apply weight tyes
  total_params - sum(p.numel() for p in model.out_head.parameters())
)


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

 tokenizer = tk.get_encoding("gpt2")
 #encode a piece of text to illustrate the generation
 start_context = "Hello, I am"
 #encode the text
 encoded = tokenizer.encode(start_context)
 print("encoded:", encoded)
 encoded_tensor = torch.tensor(encoded).unsqueeze(0) #add a batch dim
 print("encoded_tensor.shape:", encoded_tensor.shape)

 model.eval()#disable dropout since we are not training the model
 out = generate_text_simple( ##call the generate_text_simple method to iterate through the tokens and try to predict the next token
 model=model,
 idx=encoded_tensor,
 max_new_tokens=6,
 context_size=cfg["context_length"]
)
 print("Output:", out)
 print("Output length:", len(out[0]))
 #decode the text
 decoded_text = tokenizer.decode(out.squeeze(0).tolist())
 print(decoded_text)

 

main()




