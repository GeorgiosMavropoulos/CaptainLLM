"""This file contains a simple method which feeds text to the model, and the model tries to predict whether it's spam or not"""

import torch
class ReviewClassifierModel:
    pass

  #method to review the model's prediction
    def classify_review(model,text,tokenizer, device, max_length=None,pad_token_id=50256):
     model.eval() #set the model to evaluation mode

     #tokenize texts
     input_ids = tokenizer.encoder(text)
     
     supported_context_length = model.pos_emb.weight.shape[1]

     input_ids = input_ids[:min( #truncate sequences if they are too long
max_length, supported_context_length
)]
     input_ids += [pad_token_id] * (max_length - len(input_ids)) #pad sequences to the longest sequence

     #convert to tensor
     input_tensor = torch.tensor(input_ids,dtype=torch.long,device=device).unsqueeze(0) 
    
    


     with torch.no_grad(): #do not apply any gradient
        logits = model(input_tensor)[:, -1, :] #get the last values from the transformers block
     predicted_label = torch.argmax(logits, dim=-1).item()

     return "spam" if predicted_label == 1 else "not spam"

