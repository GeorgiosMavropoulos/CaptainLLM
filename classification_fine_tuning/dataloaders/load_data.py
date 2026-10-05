### this file contains the code to load the data from the training batches
from torch.utils.data import DataLoader
import torch
from ..SplitData.prepare_data import PrepareTraining 
class LoadData:
    def __init__(self):
     pass
    
    prepared_data = PrepareTraining()
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)


    train_loader = DataLoader(
    dataset=prepared_data.train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    )
    val_loader = DataLoader(
    dataset=prepared_data.val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
    )
    test_loader = DataLoader(
    dataset=prepared_data.test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    )


    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")