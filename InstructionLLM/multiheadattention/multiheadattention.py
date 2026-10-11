###IMPLEMENT THE MULTIHEAD ATTENTION CLASS WHICH CREATES MULTIPLE INSTANCES OF THE CASUAL ATTENTION CLASS

import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    #initialize the constructor
    #num heads defines how many CasualAttention instances we create
    def __init__(self,d_in, d_out, context_length,dropout, num_heads,qkv_bias=False):
        super().__init__()
        #assert the the features are integers since we want 3 columns, not 3.3
        assert (d_out % num_heads == 0), \
"d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads   ##perform an division between dimension out put and num_heads

        #initialize linear layer weigth's values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) ##this calculate the last trained transformation of the output

        self.dropout = nn.Dropout(dropout) ##drop out some columns to avoid over fitting

        self.register_buffer( ##transform the matrix with ones (implement the mask to cover the next token)
        "mask",
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
        )

        
    #use the forward class to create the weigths
    def forward(self, x):
        b, num_tokens, d_in = x.shape #get batch size, number of tokens and dimension input from the input tokens
        #iterate the input tokens from 3 different linear layers
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # Reshape keys: [batch, tokens, d_out] -> [batch, tokens, heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # Reshape Values: [batch, tokens, d_out] -> [batch, tokens, heads, head_dim]
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
         # Reshape queries: [batch, tokens, d_out] -> [batch, tokens, heads, head_dim]
        queries = queries.view( b, num_tokens, self.num_heads, self.head_dim)
        #view changes x's shape without chaning it's data
        #example
        #x.shape = [2, 6, 8]
        #after view() = (2, 6, 2, 4), this happend because we have 2 heads and 8 x 2 = 4
        ############################
        #create the transpose matrices by changing the dimension one with the dimension 2
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #truncate mask with the number of tokens
        attn_scores = queries @ keys.transpose(2,3) #calculate the attention scores by multiplying queries with the transpose of keys
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]##create the casual mask

        attn_scores.masked_fill_(mask_bool, -torch.inf) #use the mask to fill attention scores. we create negative values into the masked attn scores since pytorch reads them as 0s

        ##apply the softmax to normalize the weights
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)##this function divides attention scores with the head_dim since keys.shape = [batch, heads, tokens, head_dim] and shape-1 = head_dim
        attn_weights = self.dropout(attn_weights)## dropout attention weigths

        #create the context vector by multiplying the attn_weigths with values matrix and defining it's transpose
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #add one more layer of projection
        return context_vec


    
    

