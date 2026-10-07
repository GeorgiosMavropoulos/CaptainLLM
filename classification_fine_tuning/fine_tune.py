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
      num_epochs =6

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
tests_normal = [    ("Hey, are we still meeting for lunch today?"),
    ("I'll call you when I get home."),
    ("Don't forget to bring your passport tomorrow."),
    ("Can you send me the notes from today's lecture?"),
    ("Happy birthday! Hope you have a great day."),
    ("The meeting has been moved to 3 PM."),
    ("Thanks for your help yesterday, I really appreciate it."),
    ("Your package has been delivered to the front door."),

  
    ("Congratulations! You have won £1,000. Call now to claim your prize."),
    ("WIN a brand new phone! Text WIN to 80085 now!"),
    ("FREE cash reward waiting for you! Claim now!"),
    ("You have been selected for a £500 prize. Call now!"),
    ("Get FREE entry into our weekly cash draw. Reply WIN now!"),
    ("URGENT! Claim your guaranteed cash reward today!"),
    ("You've WON! Call 09012345678 to receive your prize."),
    ("Exclusive offer! Get £500 cash today. Apply now!"),]



tests_medium = [
   
    ("Your electricity bill has been paid successfully."),
    ("Your appointment is confirmed for Monday at 10 AM."),
    ("Your order is ready for collection. Please bring your confirmation."),
    ("Your mobile plan has been renewed successfully."),
    ("Your card payment of £45.20 was successfully processed."),
    ("Your application has been received. We will contact you shortly."),
    ("Your account statement is now available. Please log in to view it."),
    ("Your monthly subscription payment has been received. Thank you."),

    
    ("Your account qualifies for an exclusive cash offer. Apply today."),
    ("You are eligible for a fast cash loan. Apply now."),
    ("Get an instant cash advance with no waiting. Apply today!"),
    ("Your credit limit can be increased immediately. Click to apply."),
    ("Limited time offer! Get approved for a personal loan today."),
    ("You have been pre-approved for £5,000. Apply now to receive funds."),
    ("Special financial offer available now. Contact us to claim."),
    ("Need cash urgently? Apply now for an instant loan."),
]


tests_hard = [
    
    "Your loan application has been approved. Please contact your bank advisor.",
    "Your mortgage application has been approved. Please speak with your advisor.",
    "Your university application has been approved. Please check your email.",
    "Your insurance claim has been approved. We will contact you shortly.",
    "Your credit card application was successful. Your card will arrive soon.",
    "Your scholarship application has been approved. Check the student portal.",
    "Your salary advance request has been approved. Contact HR for details.",
    "Your refund has been approved and will be credited to your account.",

    
    "Your loan has been approved! Click here to receive your money.",
    "Your loan application has been approved. Claim your funds today.",
    "Your credit application was successful. Apply now to access your money.",
    "You have been approved for a personal loan. Contact us today.",
    "Your credit has been approved. Visit the link below to receive your funds.",
    "Congratulations! Your application has been approved. Claim your cash now.",
    "Your application was successful! Receive your guaranteed cash today.",
    "You qualify for £5,000 cash. Complete your application immediately.",
]

tests_confusing = [
    ("Your account has been updated. Please review your latest statement."),
    ("Your payment has been received. No further action is required."),
    ("Your bank appointment is confirmed for tomorrow at 2 PM."),
    
    ("You've been selected for a £2,000 cash reward. Claim it now."),
    ("Your account is eligible for an exclusive cash bonus. Apply today."),
    ("Act now to receive your guaranteed £5,000 payment."),
]

for t in tests_confusing:
    print(FineTune.test_classification(t), "|", t[:60])
   
    

    
    
     

   


    
    
    
   