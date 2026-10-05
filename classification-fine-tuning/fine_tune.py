###This is the main file where I load the model, load the pretrained weigths and fine tune the model
from gptmodel.config import GPT_CONFIG_124M as cfg
from download_datasets.gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from load_weigths.load_weigths import LoadWeigths
from train_script.generation import TextGeneration
from tokenizer_bpe.__tokenizer import _Tokenizer
from train_script.calculate_loss_functions import CalculateLoss
from train_script.train import Train
from calculate_accuracy_loss.calculate_accuracy_loader import CalcAccuracy
from SplitData.prepare_data import PrepareTraining
from dataloaders.load_data import LoadData
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

    #initialize calculate accuracy class
    calc_accuracy = CalcAccuracy()

    #initialiaze an instance of load data class
    load_data = LoadData()

    def config_model():

        model_configs = { ##available configurations for the GPT2 Model
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        gpt2_small_model = "gpt2-small (124M)"
        input = "Every effort moves"
        
        cfg.update(model_configs[gpt2_small_model]) #update configurations for the gpt2_small
        
        return gpt2_small_model, cfg

    #call config model method
    gpt2_small_model,cfg = config_model()
    
    model_size = gpt2_small_model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
    )
    model = GPTModel(cfg)
    load_weigths.load_weights_into_gpt(model, params)
    model.eval()


     #update model's head output, since we want to output 2 tokens (o for ham and 1 for spam)
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
    in_features=cfg["emb_dim"],
        out_features=num_classes
        )

   
    #train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
     
    #freeze model's parameters, since it's not necessary to train all the parameters
    for param in model.parameters():
     param.requires_grad = False

    # Unfreeze only the last Transformer block
    for param in model.trf_blocks[-1].parameters():
     param.requires_grad = True

    # Unfreeze only the final normalization layer
    for param in model.final_norm.parameters():
     param.requires_grad = True


    inputs = tokenizer.encoder("You won 3000 euross")
    inputs = torch.tensor(inputs).unsqueeze(0)
   
    with torch.no_grad():
        outputs = model(inputs)
      
    logits = outputs[:, -1, :]
    label = torch.argmax(logits) #computing the token with the highest probability score

    #calculate training accuracy
    train_accuracy = calc_accuracy.cacl_accuracy_loader(load_data.train_loader,model,device,num_batches=10)

    #calculate validation accuracy
    validation_accuracy = calc_accuracy.cacl_accuracy_loader(load_data.val_loader,model,device,num_batches=10)

    #calculate test accuracy
    testing_accuracy = calc_accuracy.cacl_accuracy_loader(load_data.test_loader,model,device,num_batches=10)

    #print accuracy
    print(f"Train accuracy:{train_accuracy*100:.2f}%")
    print(f"Validation accuracy:{validation_accuracy*100:.2f}%")
    print(f"Testing accuracy:{testing_accuracy*100:.2f}%")

    
     

   


    
    
    
   