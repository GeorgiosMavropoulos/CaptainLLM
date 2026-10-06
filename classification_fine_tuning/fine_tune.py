###This is the main file where I load the model, load the pretrained weigths and fine tune the model
from gptmodel.config import GPT_CONFIG_124M as cfg
from download_datasets.gpt_download import download_and_load_gpt2
from gptmodel.gpt_model import GPTModel
from load_weigths.load_weigths import LoadWeigths
from .classify_review.classify_review import ReviewClassifierModel
from tokenizer_bpe.__tokenizer import tokenizer
from .SplitData.prepare_data import PrepareTraining
from .training_script.training_classifier import Trainer
import time
from .dataloaders.load_data import LoadData
from .calculate_loss.calculateloss import CalculateClassificationLoss
import torch
import matplotlib.pyplot as plt

class FineTune:
    def __init__(self):
      pass

    #initialize LoadWeigths class
    load_weigths = LoadWeigths()

    #initialiaze an instance of load data class
    load_data = LoadData()

    #create an instance of CalculateLoss
    calculate_loss = CalculateClassificationLoss()

    #create an instance of the trainer class
    trainer = Trainer()

    #create an instance of the classify review class
    classify_reviewer = ReviewClassifierModel

    prepare_data = PrepareTraining()

    #call config model method
    gpt2_small_model = GPTModel(cfg)
    model_size="124M"
    
    settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
    )

    
    
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
    print(device)
        
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

    for param in gpt2_small_model.out_head.parameters(): #unfreeze the new outhead
        param.requires_grad = True



     #function to validate the model if it can actually classify
   
    def test_classification(text):
             
             answer = FineTune.classify_reviewer.classify_review(
             FineTune.gpt2_small_model,
             text,
             tokenizer,
             device=FineTune.device,
             max_length=FineTune.prepare_data.train_dataset.max_length,
         )
             return answer

         
            

    #create a training loop
    @staticmethod
    def train(model,train_loader,val_loader,device):
      start_time = time.time()
      
      # DEFINE the optimizer and delegate it into a variable
      optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
      num_epochs = 15

      #train the model
        #load the pretrained model
      checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)

      model.load_state_dict(
      checkpoint["model_state_dict"]
)

      optimizer.load_state_dict(
     checkpoint["optimizer_state_dict"]
)
     
      train_losses, val_losses, train_accs, val_accs, examples_seen = \
      FineTune.trainer.train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50,eval_iter=15)
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

      #save the model's and optimizer's state
      torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": num_epochs,
}, "checkpoint.pth") 

  


    
        

#execute train function
FineTune.train(
    FineTune.gpt2_small_model,
    FineTune.load_data.train_loader,
    FineTune.load_data.val_loader,
    FineTune.device
)
#test if classification actually works
tests = [
    "Congratulations!!!! I was sure that you are going to win the lottary Trevor,",

    "Hey, are we still meeting for coffee at 6 tonight?",

    "Congratulations! You have won a FREE prize! Call now to claim your reward!",

    "Can you send me the notes from today's lecture?",

    "URGENT! You have been selected to receive £5000. Reply YES to claim.",

    "Happy birthday! Hope you have an amazing day!",

    "You won a brand new iPhone! Click here now to collect your prize.",

    "I'll be home around 8, do you need me to pick up anything?",

    "Thanks for helping me with the project yesterday.",

    "FREE entry to win £1000 cash! Text WIN to 80082 now!",

    "Are you free this weekend? I was thinking we could go for a walk.",

    "Congratulations! Your mobile number has won a cash prize of $500.",

    "Don't forget to bring your charger tomorrow.",

    "Exclusive offer! Get 90% OFF all products today only. Buy now!",

    "Hey, can you call me when you finish work?",

    "You have been chosen for a special reward. Call 09061234567 to claim.",

    "I left your keys on the kitchen table.",

    "WINNER! You have won a luxury holiday. Reply CLAIM to receive it.",

    "Good luck with your exam tomorrow! You got this.",

    "Limited time offer! Get cheap loans with no credit check. Apply now!",

    "You've received a bonus of £250. Verify your account immediately to collect it."
]
for t in tests:
    print(FineTune.test_classification(t), "|", t[:60])
   
    

    
    
     

   


    
    
    
   