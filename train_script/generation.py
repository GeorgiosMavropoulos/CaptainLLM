### this class contains the generate text function and prints the text
import torch

class TextGeneration:
    
   #predict the next token to generate text
      #idx is a batch (batch,n_tokens) array of indices in the current context
    @staticmethod
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



        #evaluate the progress by printing a text after each epoch
    @staticmethod
    def generate_and_print_sample(model, tokenizer, device, start_context):
       model.eval()
       context_size = model.pos_emb.weight.shape[0] #access positional embedding's weights
       encoded = torch.tensor(tokenizer.encoder(start_context),dtype=torch.long).unsqueeze(0).to(device)
       with torch.no_grad(): ##do not apply gradients
            #use the generate_text_simple method
            token_ids = TextGeneration.generate_text_simple(model=model, idx=encoded,max_new_tokens=50, context_size=context_size)
            #decode the returned text
       decoded_text = tokenizer.decoder(token_ids[0].tolist())
       print(decoded_text.replace("\n", " "))#remove any unwanted symbols
       #start training
       model.train()
          