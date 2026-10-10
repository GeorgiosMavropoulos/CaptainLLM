"""This file contains the function which splits the given dataset """
from datasets.download_dataset_with_instructions import data 

train_dataset = int(len(data) * 0.85) #keep the 85% of the text as training dataset
test_portion = int(len(data) * 0.1) #use 10% for testing
val_portion = len(data) - train_dataset - test_portion #use the rest 5% for validation

train_data = data[:train_dataset]
test_data = data[train_dataset:train_dataset + test_portion]
val_data = data[train_dataset + test_portion:]


