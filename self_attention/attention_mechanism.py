###this file contains the attention class
import torch.nn as nn
import torch

class SelfAttentionV1(nn.Module):
    #initialize the constructor with key, query, values
    def __init__(self,d_in,d_out,qkv_bias=False):
        ###initialize the trainable weight matrices for queries, keys, and values, each of them transforimg the input dimension (d_in) into an output dimension (d_out) 
        ##nn.linear is a better way to initialize with random values, since it has an optimized weight initialization scheme, contributing to more stable and effective model training.
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    #create the method which computes the attention weigths
    def forward(self,x):
        ##calculate the dot product of keys, values and the query
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #calculate the attention scores
        attention_scores = queries @ keys.T
        #normalize scores using softmax
        attention_weigths = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        #calculate the context vector
        context_vector = attention_weigths @ values

        return context_vector

