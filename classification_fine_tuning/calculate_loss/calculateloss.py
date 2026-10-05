"""This class contains the code to calculate loss in accuracy using the cross entropy method"""
import torch
class CalculateClassificationLoss:
    def __init__(self):
        pass


    #calculate loss method
    #create a function to calculate a cross entropy loss of a given batch
    @staticmethod
    def calc_loss_batch(input_batch, target_batch, model, device):
           
            """ The transfer to a given device allows us to transfer the data to a GPU."""
           
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)[:, -1, :] #calculate the logits from the input batch
            #calculate the loss
            loss = torch.nn.functional.cross_entropy(logits, target_batch)
            return loss



    
          #function to calculate the loss of all given batches from the data loader
    @staticmethod
    def calc_loss_loader(data_loader, model, device, num_batches=None):
            total_loss = 0
            #return an error message if no data exists
            if len(data_loader) == 0: 
              return float("nan")
            #iterate through all batches if num_batches has not being given
            elif num_batches is None:
              num_batches = len(data_loader)
              """else block reduces the number of batches to match the total number of batches in the data loader if num_batches exceeds the number of batches in the data loader"""
            else:
              num_batches = min(num_batches, len(data_loader))
            for i, (input_batch, target_batch) in enumerate(data_loader):
             
             if i < num_batches:
              #calculate the loss of each batch
              loss = CalculateClassificationLoss.calc_loss_batch(input_batch, target_batch, model, device)
              total_loss += loss.item() #summarize the loss of each batch
             else:
              break
            return total_loss / num_batches ##return the average loss  of all batches if i == num_batches