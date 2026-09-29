
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
import torch
from gptmodel.gpt_model import  TransformerBlock
import tiktoken as tk
from dataloader.windowslider import LoadData
##test DataLoad from pytorch
def main():
 
 #create an instance of LoadData class
 data_loader = LoadData()

 #create an instance of GPTModel class
 model = GPTModel(cfg)
 
 #let's prepare the data
 def load_text(): 
 
  file_path = "C:/Users/Overkill/Desktop/train-llm/BaseLLM/datasets/the_verdict.txt"
  with open(file_path, "r", encoding="utf-8") as file:
   text_data = file.read()
  return text_data

  # Create the GPT-2 tokenizer
 tokenizer = tk.get_encoding("gpt2")

  #call the load data method
 dataset = load_text()
 #get total chars
 total_characters = len(dataset)
 #get total tokens
 total_tokens = len(tokenizer.encode(dataset))
 #print("Characters:", total_characters)
 #print("Tokens:", total_tokens)

 train_ratio = 0.90
 split_idx = int(train_ratio * len(dataset)) #multiply train ratio with dataset's length to get the 90% of the text
 train_data = dataset[:split_idx] #split the remain 10%
 val_data = dataset[split_idx:]
 #split data
 #we will use a ration of 90% for training data and 10% for validation data

 #define a method which splits the dataset into trainable and validation data
 def split_dataset(train_data,val_data):

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
  #return f"Data was splitted with success"

 trainable_data,validation_data =split_dataset(train_data,val_data)

 #create a function to calculate a cross entropy loss of a given batch
 def calc_loss_batch(input_batch, target_batch, model, device):
 
  """ The transfer to a given device allows us to transfer the data to a GPU."""
 
  input_batch = input_batch.to(device)
  target_batch = target_batch.to(device)
  logits = model(input_batch) #calculate the logits from the input batch
  #calculate the loss
  loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
  return loss


 #function to calculate the loss of all given batches from the data loader
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
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    total_loss += loss.item() #summarize the loss of each batch
   else:
    break
   return total_loss / num_batches ##return the average loss  of all batches if i == num_batches

 #if a cuda gpu is available train the llm on cuda, otherwise on the cpu
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model.to(device) ##force model to train on the available device
 with torch.no_grad(): #disable gradient for efficiency since we are not training yet
  train_loss = calc_loss_loader(trainable_data, model, device)
  val_loss = calc_loss_loader(validation_data, model, device)

 #print losses
 print("Training loss:", train_loss)
 print("Validation loss:", val_loss)

 




"""
 ##test the gpt model
 torch.manual_seed(123)

 batch = torch.randint(
    0,
    cfg["vocab_size"],
    (2, 4)
)

 model = GPTModel(cfg) #instanciate a model using the gpt model with gpt_config_124m


 #out = model(batch)
 
 #predict the next token to generate text
 #idx is a batch (batch,n_tokens) array of indices in the current context
 def generate_text_simple(model,idx,max_new_tokens,context_size):
  for _ in range(max_new_tokens):
   idx_cond = idx[:, -context_size:] ##abstract the context_size from the idx, in order not to outnumber the given context_length
   with torch.no_grad(): ##do not apply gradient
    logits = model(idx_cond) #create the logits

   logits = logits[:, -1, :] #focuses only on the last time step, so that (batch, n_tokens, vocabulary_size) becomes (batch, vocabulary_size)
   #convert the tokens to a propability distribution using softmax
   propabilities = torch.softmax(logits, dim=-1)
   #calculate the next token by finding the highest propability token using torch.argmax
   idx_next = torch.argmax(propabilities, dim=-1, keepdim=True)
   idx = torch.cat((idx, idx_next), dim=1) #append the sampled index to the running sequence, where idx has shape (batch, n_tokens +1)

  return idx #idx has shape (batch,1)

 ##training loss calculation
 def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

 def token_ids_to_text(token_ids, tokenizer):
  flat = token_ids.squeeze(0)
  return tokenizer.decode(flat.tolist())

 start_context = "Every effort moves you"
 tokenizer = tk.get_encoding("gpt2")

 token_ids = generate_text_simple(
 model=model,
 idx=text_to_token_ids(start_context, tokenizer),
 max_new_tokens=10,
 context_size=cfg["context_length"]
)
 #print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#create the inputs into torch tensor
 #inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
#[40, 1107, 588]]) #"I really like"]

 #targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
#[1107, 588, 11311]]) # " really like chocolate"]

 #with torch.no_grad():#do not apply gradient since we are not training yet 
  #logits = model(inputs) #create the logits
 #probas = torch.softmax(logits, dim=-1) #propability of each token in vocabulary
 #print(probas.shape)

# token_ids = torch.argmax(probas, dim=-1, keepdim=True) #calculate the token ids with the highest propability score

 #print propability scores for each text
 #text_idx = 0
 #target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
 #print("Text 1:", target_probas_1)
 #text_idx = 1
 #target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
 #print("Text 2:", target_probas_2)

 #create the logarithm based on propabilites. we concatenate the two torches
 #log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
 #print(log_probas)

 #find the average of log_propabilities
 #avg_log_probas = torch.mean(log_probas)
 #print(avg_log_probas)

 #convert the negative number occured from the mean of probabilities into a positive number (cross entropy loss)
 #neg_ang_log_probas = avg_log_probas * -1
 #print(neg_ang_log_probas)

 #use cross entropy function to avoid all these steps
 #get logits and target token's shape
 #print("Logits shape:", logits.shape)
 #print("Targets shape:", targets.shape)

#mock some vectors
 inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
 [40, 1107, 588]]) #"I really like"]
 
 targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
 [1107, 588, 11311]]) # " really like chocolate"]

 with torch.no_grad():#do not apply gradient since we are not training yet 
   logits = model(inputs) #create the logits

 #flatten the tensors before apply the cross entropy function
 logits_flat = logits.flatten(0, 1)
 targets_flat = targets.flatten()
 #print("Flattened logits:", logits_flat.shape)
 #print("Flattened targets:", targets_flat.shape)

 #calculate the loss using cross entropy function
 #using cross entropy we save so many lines of code

 
 loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
 print(loss)
 perplexity = torch.exp(loss)
 print(perplexity)

"""

 

main()




