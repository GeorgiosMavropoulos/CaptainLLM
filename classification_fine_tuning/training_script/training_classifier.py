"""This file contains the main training and evaluation functions"""
from ..calculate_loss.calculateloss import CalculateClassificationLoss
from ..calculate_accuracy_loss.calculate_accuracy_loader import CalcAccuracy
import torch
class Trainer:
    def __init__(self):
      pass
    
    
     #Initialize lists to track losses andexamples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1


    #initialize an instance of calculate loss class
    calc_loss = CalculateClassificationLoss()

    #create an instance of calculate accuracy class
    calc_accuracy = CalcAccuracy()

    #training function
    @staticmethod
    def train_classifier(model, train_loader, val_loader, optimizer, device,num_epochs, eval_freq, eval_iter):
        ##main training loop which trains the model based on the given epochs number
        for i in range(num_epochs):
            model.train()

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() #reset loss gradients from the previous batch iteration
                loss = Trainer.calc_loss.calc_loss_batch(input_batch, target_batch, model, device) #calculate the loss per batch
                loss.backward() #apply backpropagation to calculate loss gradient
                optimizer.step() #update model's weigths based on the computed loss gradient
                examples_seen += input_batch.shape[0] #calculate how many examples the model saw
                global_step += 1

                if global_step % eval_freq == 0: #evaluation step 
                    train_loss, val_loss = Trainer.evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                    Trainer.train_losses.append(train_loss)
                    Trainer.val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                    )


            #calculate train and validation accuracy
            train_accuracy = Trainer.calc_accuracy.calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
            )
            val_accuracy = Trainer.calc_accuracy.calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
            )

            #print training and validation accuracy
            print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy*100:.2f}%")
            ##append train's accuracy and val's accuracy values to the empty lists
            Trainer.train_accs.append(train_accuracy)
            Trainer.val_accs.append(val_accuracy)

        return Trainer.train_losses, Trainer.val_losses, Trainer.train_accs, Trainer.val_accs, examples_seen


    #evaluate model function
    @staticmethod
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
         model.eval()
         with torch.no_grad():
             train_loss = Trainer.calc_accuracycalc_loss_loader(
             train_loader, model, device, num_batches=eval_iter
             )
             val_loss = Trainer.calc_accuracycalc_loss_loader(
             val_loader, model, device, num_batches=eval_iter
             )
             model.train()
         return train_loss, val_loss