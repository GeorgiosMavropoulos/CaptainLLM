##this class loads the given text and splits the data set into trainable data and validation data
import torch
from gptmodel.config import GPT_CONFIG_124M as cfg
from dataloader.windowslider import LoadData
class PrepareData:

    #let's prepare the data
    @staticmethod
    def load_text(file_path:str): 
         
     with open(file_path, "r", encoding="utf-8") as file:
      text_data = file.read()
     return text_data

   

    #define a method which splits the dataset into trainable and validation data
    @staticmethod 
    def split_dataset(train_data,val_data):
    
      #create an instance of LoadData class
      data_loader = LoadData()
    
            ##now using train_data val_data we can create the dataset loader
      torch.manual_seed(123) #manual nums to create the trainable weigths
      #create the train loader
      train_loader = data_loader.create_dataloader_v1(
      train_data,
      batch_size=2,
      max_length=cfg["context_length"],
      stride=cfg["context_length"],
      drop_last=True,
      shuffle=True,
      num_workers=0
      )
      #define the validation loader
      val_loader = data_loader.create_dataloader_v1(
      val_data,
      batch_size=2,
      max_length=cfg["context_length"],
      stride=cfg["context_length"],
      drop_last=False,
      shuffle=False,
      num_workers=0
      )
      return train_loader, val_loader



    #method to prepare the data for the training
    @staticmethod
    def prepare_data_for_training(filepath, train_ratio):

        # Load text
        dataset = PrepareData.load_text(filepath)

        # Split text into train/validation
        split_idx = int(train_ratio * len(dataset))

        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]

        # Create dataloaders
        train_loader, val_loader = PrepareData.split_dataset(
            train_data,
            val_data
        )

        return train_loader, val_loader