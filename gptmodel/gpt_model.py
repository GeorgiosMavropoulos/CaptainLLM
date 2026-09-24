#### Implement the GPT model's architecture
import torch
import torch.nn as nn


class GPTModel(nn.Module):
    ## initialize the constructor
    def __init__(self, cfg):
        super().__init__()
        ## get vocabulary's size and embedding dims from config file
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # create the positional embeddings taking context_length and embedding dim's values
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        ## add dropout rate
        self.drop_embs = nn.Dropout(cfg["drop_rate"])
        # use a placeholder for transformer blocks
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # use the real LayerNormalization class for the final norm
        self.final_norm = LayerNormalization(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    ## set up the forward function to designate the path the data follow in the neural network
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  ## designate batch size by input's shape
        tok_embeds = self.tok_emb(in_idx)  ## compute token embeddings
        # compute positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  ## add token with positional embeddings
        x = self.drop_embs(x)  ## drop randomly some embeddings
        x = self.trf_blocks(x)  ## process the data through the transformer blocks
        x = self.final_norm(x)  ## apply the final normalization
        logits = self.out_head(x)  ## delegate into logits the linear output layer
        return logits  # return the logits


## create the (placeholder) transformer block
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):  ## this block just returns its input
        return x


## create the final layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # eps is a small constant (epsilon) added in order to prevent division by zero during the normalization
        self.eps = 1e-5
        # scale and shift are two trainable parameters of the same dimension as the input
        # so the model automatically adjusts during training to improve performance on its training task.
        # This allows the model to learn appropriate scaling and shifting that best suit the data it is processing.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    ##mean must be 0 and variance 1. this keeps the values stable. since the values pass through different layers
    #with 0 mean and var 1 we keep the data stable and the model can learn to make better predictions from this
    #data
    def forward(self, x):
        # this normalization is applied on the last dimension of x, which is the embeddings dimension
        mean = x.mean(dim=-1, keepdim=True)  # find the mean value
        # find the variance (divide by n, not n-1, i.e. biased variance)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # find the norm
        return self.scale * norm_x + self.shift