### this class contains the generate text function and prints the text
import torch

class TextGeneration:
    
   #predict the next token to generate text
      #idx is a batch (batch,n_tokens) array of indices in the current context
    @staticmethod
    def generate_text(model,idx,max_new_tokens,context_size,temperature=0.0, top_k=None, eos_id=None):
       for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] ##abstract the context_size from the idx, in order not to outnumber the given context_length
        with torch.no_grad(): ##do not apply gradient
          logits = model(idx_cond) #create the logits
  
        logits = logits[:, -1, :] #focuses only on the last time step, so that (batch, n_tokens, vocabulary_size) becomes (batch, vocabulary_size)

         #Filters logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k) #force torch to keep the top_k higherst propability tokens
            min_val = top_logits[:, -1] #take the minimum high propability token and apply it as a limit. nothing else lower than this will stay

            #filter logits. if logit is < min_val make it -inf
            ## After softmax, these tokens will have probability 0.
            logits = torch.where(logits < min_val,torch.tensor(float('-inf')).to(logits.device),logits )

         #apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature #divide logits with temperature's value

            #convert the tokens to a propability distribution using softmax
            propabilities = torch.softmax(logits, dim=-1)        
            #multimonial does not always select the token with the highest probability. applying randomness is a better approach since the model is not so much biased
            idx_next = torch.multinomial(propabilities, num_samples=1)
            ##do not allow negative temp values
        elif temperature < 0.0:
            raise ValueError("Temperature must be >= 0")
        #Carries out greedy nexttoken selection as before when temperature scaling is disabled
        else:
         #calculate the next token by finding the highest propability token using torch.argmax
         idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        #Stops generating early if end-of-sequence token is encountered
        if eos_id is not None and idx_next.item() == eos_id:
            break
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
            token_ids = TextGeneration.generate_text(model=model, idx=encoded,max_new_tokens=15, context_size=context_size,temperature=1,top_k=0)
            #decode the returned text
       decoded_text = tokenizer.decoder(token_ids[0].tolist())
       print(decoded_text.replace("\n", " "))#remove any unwanted symbols
       #start training
       model.train()
          