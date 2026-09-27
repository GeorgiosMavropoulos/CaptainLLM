
from gptmodel.gpt_model import GPTModel
from gptmodel.config import GPT_CONFIG_124M as cfg
import torch
from gptmodel.gpt_model import  TransformerBlock

##test DataLoad from pytorch
def main():


 ##test the gpt model
 torch.manual_seed(123)
 batch = torch.randint(
    0,
    cfg["vocab_size"],
    (2, 4)
)
 model = GPTModel(cfg) #instanciate a model using the gpt model with gpt_config_124m

 #delegate into a variable the output context vector
 out = model(batch)
 print("Input batch:\n", batch)
 print("\nOutput shape:", out.shape)
 print(out)

 #get model's total paremeters
 total_params = sum(p.numel() for p in model.parameters())
 print(f"Total number of parameters: {total_params:,}")

 total_params_gpt2 = (
  ##apply weight tyes
  total_params - sum(p.numel()
  for p in model.out_head.parameters())
)
 print(f"Number of trainable parameters "
 f"considering weight tying: {total_params_gpt2:,}"
)
 

main()




