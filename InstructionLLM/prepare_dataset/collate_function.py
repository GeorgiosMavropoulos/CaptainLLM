"""This module contains the collate functions. 
This function is responsible to pad the training examples in each batch to the same length while allowing different batches to have different lengths
"""
import torch

class Collate:
    def __init__(self):
        pass


    def custom_collate_fn(batch,pad_token_id=50256,device="cpu",ignore_index=-100,allowed_max_length=None,):
        #find the longest sequence on each batch
        batch_max_length = max(len(item)+1 for item in batch)
        inputs_lst, targets_lst = [], []

        #pad the prepared inputs
        for item in batch:
         new_item = item.copy()
         new_item += [pad_token_id]

         padded = (
        new_item + [pad_token_id] *
        (batch_max_length - len(new_item))
        )

        #remove the extra padded token which was added previously
         inputs = torch.tensor(padded[:-1])
         targets = torch.tensor(padded[1:]) #shift +1 to the right targets

         
         mask = targets == pad_token_id #create a boolean tensor to define the indexes which contain the padded token id
         indices = torch.nonzero(mask).squeeze() #find the indexes containing the padded token
         #get all indexes except of the first and add the ignore index (-100) to replace the padded token. I live the first token, since 50256 may exist on the actual text
         if indices.numel() > 1: 
          targets[indices[1:]] = ignore_index

         #define the check in order to restrict the max_length. Our model handles only 1024 context_length. Added this to avoid errors if the dataset is bigger than 1024
         if allowed_max_length is not None:
          inputs = inputs[:allowed_max_length]
          targets = targets[:allowed_max_length]


         inputs_lst.append(inputs)
         targets_lst.append(targets)
        inputs_tensor = torch.stack(inputs_lst).to(device) #convert the list of inputs to a tensor and transfer it to the device
        targets_tensor = torch.stack(targets_lst).to(device)
        return inputs_tensor,targets_tensor


    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1,inputs_2,inputs_3)
    inputs,targets =custom_collate_fn(batch)
    print(inputs)
    print(targets)
    