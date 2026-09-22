###this file contains the attention class
import torch.nn as nn
import torch

class CasualAttentionV1(nn.Module):
    #initialize the constructor with key, query, values
    def __init__(self,d_in,d_out,context_length, dropout, qkv_bias=False):
        ###initialize the trainable weight matrices for queries, keys, and values, each of them transforimg the input dimension (d_in) into an output dimension (d_out) 
        ##nn.linear is a better way to initialize with random values, since it has an optimized weight initialization scheme, contributing to more stable and effective model training.
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #initialize the dropout to use it to zero some elements in the table in order to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        #implement the mask in order to create zeros in the diagonial and unmask one by one elements in each row
        """
        buffers are automatically moved to the appropriate device (CPU or GPU) along with our model, which will
        be relevant when training our LLM. This means we don’t need to manually ensure
        these tensors are on the same device as your model parameters, avoiding device mismatch
        errors.
        """
        self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
        )

    #create the method which computes the attention weigths
    def forward(self,x):
        ##calculate the dot product of keys, values and the query
        b,num_tokens, d_in = x.shape #transpose dimension 1 and 2 and keep the batch dimension at 1st position
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #calculate the attention scores
        attention_scores = queries @ keys.transpose(1, 2)

        ##mask the attention socres and replace 1s with negative prices since pytorch considers them as zeros
        attention_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        #normalize scores using softmax
        attention_weigths = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        #dropout random columns to avoid overfitting
        attention_weigths = self.dropout(attention_weigths)
        #calculate the context vector
        context_vector = attention_weigths @ values

        return context_vector

