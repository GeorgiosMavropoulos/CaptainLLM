
from gptmodel.gpt_model import GPTModel,LayerNormalization,FeedForward, MockedDeepNeuralNetwork
from gptmodel.config import GPT_CONFIG_124M as cfg
import torch
from gptmodel.gpt_model import  TransformerBlock

##test DataLoad from pytorch
def main():


 ##test the transformers block
 torch.manual_seed(123)
 x = torch.rand(2, 4, 768) #create a sample input shape (batch_size, num_tokens, emb_dim)
 #define the transformers block
 block = TransformerBlock(cfg)
 output = block(x)

 print("Input shape:", x.shape)
 print("Output shape:", output.shape)
 

main()




