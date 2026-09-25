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


        # The scale and shift parameters let the model partially undo the strict
        # normalization when that helps it learn. 
        # Forcing every layer's output to # have mean=0 and variance=1 keeps training stable, but it can be overly
        # restrictive: some dimensions might benefit from a different variance or
        # from not being centered at 0. Because scale and shift are nn.Parameter,
        # PyTorch registers them as trainable weights, so they participate in
        # backpropagation and get updated by the optimizer during training, letting
        # the model learn the best scale/shift per embedding dimension directly
        # from the data. They start at ones and zeros so that, at the very
        # beginning of training, scale * norm_x + shift == norm_x -- i.e. the layer
        # starts out as pure normalization with no extra effect. As training
        # proceeds, the optimizer only moves them away from 1/0 if doing so
        # actually reduces the training loss, so any adjustment is learned, not
        # hand-set, and only happens where it genuinely helps performance.

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




#implement the GELU activation
##In contrast with RELU, GELU function allow some minor negative values to pass in order create a smooth curve around Zero
# After the input pass through the normalization layer it passes also from this function so as the network gets a non-linearity.
#This helps the model learn more efficiently difficult language patterns
class GELU(nn.Module):
    def __init__(self):
     super().__init__()

    def forward(self, x): #
      # 1. x + 0.044715 * x^3: Cubic correction term used for the tanh approximation.
      # 2. torch.sqrt(2.0 / pi): Scaling factor (~0.79788) derived from the normal distribution.
      # 3. torch.tanh(...): Smoothly maps the scaled values between -1 and 1.
      # 4. 0.5 * x * (1 + tanh(...)): Final smooth activation gating mechanism.
     return 0.5 * x * (1 + torch.tanh(
     torch.sqrt(torch.tensor(2.0 / torch.pi)) *
     (x + 0.044715 * torch.pow(x, 3))
     ))


##feed forward function
class FeedForward(nn.Module):
    def __init__(self, cfg):
     super().__init__()
    ##create 2 linear layers and use a GELU function
     self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x): ##pass data through the neural network
        return self.layers(x)



#mocked deep neural network
class MockedDeepNeuralNetwork(nn.Module):
   def __init__(self, layer_sizes, use_shortcut):
      super().__init__()
      self.use_shortcut = use_shortcut
      ##implement 5 layers with GELU activation
      self.layers = nn.ModuleList([nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
GELU()),
nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
GELU()),
nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
GELU()),
nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
GELU()),
nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
GELU())])

   #forward the data into the neurla network blocks
   def forward(self, x):
      for layer in self.layers:
        layer_output = layer(x) #compute the output of the current layer
        ##if a shortcut is being used add the current' layer's embeddings into the output
        if self.use_shortcut and x.shape == layer_output.shape:
         x = x + layer_output
        else:
            x = layer_output
      return x