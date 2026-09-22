###IMPLEMENT THE MULTIHEAD ATTENTION CLASS WHICH CREATES MULTIPLE INSTANCES OF THE CASUAL ATTENTION CLASS
from self_attention import attention_mechanism
import torch.nn as nn
import torch

class MultiHeadAttentionWrapper(nn.Module):
    #initialize the constructor
    #num heads defines how many CasualAttention instances we create
    def __init__(self,d_in, d_out, context_length,dropout, num_heads,qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [attention_mechanism.CasualAttentionV1( ##call the casual attention class
            d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
            )
    #use the forward class to create the weigths
    def forward(self, x):
         #torch.cat concantenates the tensors in an existent dimension
         return torch.cat([head(x) for head in self.heads], dim=-1)