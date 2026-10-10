"""This class contains the pytorch's data loaders to create the batches, shuffle the data so the model does not memorize the sequence trasnfer the data from RAM to GPU and define workers number
since the OS supports parallelism the training may be completed quicker
"""
from prepare_dataset.collate_function import Collate
from prepare_dataset.instructions_dataset import InstructionsDataset
from prepare_dataset.split_dataset import train_data, test_data,val_data
from tokenizer_bpe.__tokenizer import _Tokenizer
tokenizer = _Tokenizer()
from functools import partial
import torch
from torch.utils.data import DataLoader
collate_fun = Collate()

class PyTorchDataLoaders:
    def __init__(self, batch_size=8, num_workers=0):
     self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     self.num_workers = num_workers
     self.batch_size = batch_size

     
     self.customized_collate_fn = partial(
            collate_fun.custom_collate_fn,
            device=self.device,
            allowed_max_length=1024
        )
    def get_loaders(self):
        #create the dataloaders
        #training dataloader
        training_dataset =  InstructionsDataset(train_data,tokenizer)
        #create the training dataloader
        train_loader = DataLoader(training_dataset,batch_size=self.batch_size,collate_fn=self.customized_collate_fn,
                                drop_last=True, #use drop_last value because the dataset may contain 1005 values but we want for instance to train 100 values x 10.
                                # In order not to create a batch with 5 values from the dataset we drop the other 5
                                num_workers=self.num_workers,
                                shuffle=True)

        validation_dataset = InstructionsDataset(val_data,tokenizer)
        #create the validation dataloader
        val_loader = DataLoader(validation_dataset,batch_size=self.batch_size,collate_fn=self.customized_collate_fn,
                                drop_last=False,  #we use drop last value = False since we want to run the validation throught all the values from the dataset
                                num_workers=self.num_workers,
                                shuffle=False #we set shuffle to false since it costs computation time and there is no feaf of the model memorizing the dataset
                                )

        #create the testing dataloader
        testing_dataset = InstructionsDataset(test_data,tokenizer)
        test_loader = DataLoader(testing_dataset,batch_size=self.batch_size,collate_fn=self.customized_collate_fn,
                                drop_last=False,  #we use drop last value = False since we want to run the validation throught all the values from the dataset
                                num_workers=self.num_workers,
                                shuffle=False #we set shuffle to false since it costs computation time and there is no feaf of the model memorizing the dataset
                                )

        #let's examine the tensors
        print("Train loader:")
        for inputs,targets in train_loader:
           print(inputs.shape,targets.shape)


        return train_loader, test_loader, val_loader

torch_loaders = PyTorchDataLoaders()
torch_loaders.get_loaders()
