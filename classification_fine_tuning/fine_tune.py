###This is the main file where I load the model, load the pretrained weigths and fine tune the model
from gptmodel.config import GPT_CONFIG_124M as cfg
from download_datasets.gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from load_weigths.load_weigths import LoadWeigths
from train_script.generation import TextGeneration
from tokenizer_bpe.__tokenizer import _Tokenizer
from .training_script.training_classifier import Trainer
import time
from .dataloaders.load_data import LoadData
from .calculate_loss.calculateloss import CalculateClassificationLoss
import torch
class FineTune:
    def __init__(self):
      pass

    #initialize LoadWeigths class
    load_weigths = LoadWeigths()

    #initialize an object from TextGeneration class
    generate_text = TextGeneration()

    #create an instance of _Tokenizer class
    tokenizer = _Tokenizer()


    #initialiaze an instance of load data class
    load_data = LoadData()

    #create an instance of CalculateLoss
    calculate_loss = CalculateClassificationLoss()

    #create an instance of the trainer class
    trainer = Trainer()

    def config_model():

        model_configs = { ##available configurations for the GPT2 Model
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        gpt2_small_model = "gpt2-small (124M)"
        
        
        cfg.update(model_configs[gpt2_small_model]) #update configurations for the gpt2_small
        
        return gpt2_small_model, cfg

    #call config model method
    gpt2_small_model,cfg = config_model()
    
    model_size = gpt2_small_model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
    )
    gpt2_small_model = GPTModel(cfg)
    load_weigths.load_weights_into_gpt(gpt2_small_model, params)
    gpt2_small_model.eval()

    #update model's head output, since we want to output 2 tokens (o for ham and 1 for spam)
    torch.manual_seed(123)
    num_classes = 2
    gpt2_small_model.out_head = torch.nn.Linear(
    in_features=cfg["emb_dim"],
        out_features=num_classes
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    gpt2_small_model.to(device)
   
     
    #freeze model's parameters, since it's not necessary to train all the parameters
    for param in gpt2_small_model.parameters():
     param.requires_grad = False

    # Unfreeze only the last Transformer block
    for param in gpt2_small_model.trf_blocks[-1].parameters():
     param.requires_grad = True

    # Unfreeze only the final normalization layer
    for param in gpt2_small_model.final_norm.parameters():
     param.requires_grad = True


    #create a training loop
    @staticmethod
    def train(model,train_loader,val_loader,device):
      start_time = time.time()
      
      # DEFINE the optimizer and delegate it into a variable
      optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
      num_epochs = 5

      #train the model
     
      train_losses, val_losses, train_accs, val_accs, examples_seen = \
      FineTune.trainer.train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50,eval_iter=10)
      end_time = time.time()
      #calculate the training time
      execution_time_minutes = (end_time - start_time) / 60
      print(f"Training completed in {execution_time_minutes:.2f} minutes.")

      test_accuracy = FineTune.trainer.calc_accuracy.calc_accuracy_loader(
                       FineTune.load_data.test_loader,
                       model,
                       device,
                       num_batches=None
                   )
       
      print(f"Test accuracy: {test_accuracy * 100:.2f}%")
          

#execute train function
FineTune.train(
    FineTune.gpt2_small_model,
    FineTune.load_data.train_loader,
    FineTune.load_data.val_loader,
    FineTune.device
)

   
    

    
    
     

   


    
    
    
   