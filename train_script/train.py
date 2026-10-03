### this file contains the training class which will be called from the main file to activate the trainign

import torch

from .calculate_loss_functions import CalculateLoss
from .generation import TextGeneration
class Train:

    #simple training function
    @staticmethod
    def train_model_simple(model, train_loader, val_loader,
        optimizer, device, num_epochs,
        eval_freq, eval_iter, start_context, tokenizer):
        train_losses, val_losses, track_tokens_seen = [], [], [] ##initialize lists to track validation/train losses and track the number of produced tokens
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs): #start the main train loop
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() #reset the loss gradients from the previous batch iteration
                #calculate the loss from the current batch
                loss = CalculateLoss.calc_loss_batch(input_batch, target_batch, model, device)
                
                loss.backward() ##calculate the loss using backpropagation
                optimizer.step() #update model weigths using loss gradient
                tokens_seen += input_batch.numel()
                global_step += 1 #increment the step 

                #optional evaluation step
                if global_step % eval_freq == 0: #evaluate the model when the training is done
                    train_loss, val_loss = Train.evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss) ##append the loss into the list with losses values
                    val_losses.append(val_loss) #append validation losses
                    track_tokens_seen.append(tokens_seen) #store the tokens into the list
                    #print the results
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {train_loss:.3f}, "
                            f"Val loss {val_loss:.3f}"
                            )
            #print a sample text after each epoch
            TextGeneration.generate_and_print_sample(model, tokenizer, device, start_context)
           

        return train_losses, val_losses, track_tokens_seen #return the losses


    #method to evaluate the model
    @staticmethod
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
       model.eval() #this mode disables dropout to stabilize the the results
       #disable gradients to calculate the actual loss
       with torch.no_grad():
            #calculate the training loss
            train_loss =  CalculateLoss.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            #calculate validation loss
            val_loss =  CalculateLoss.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
       model.train() #start training
       return train_loss, val_loss





            