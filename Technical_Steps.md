#### This file contains the technical steps I took in order to build the llm as the book says. Afterwards this report will be used to build up my own code agent.

### Dataset
1. the_verdict.txt (a short story found available on wikipedia)

### Tokenization
Tokenization is the process of seperating words, symbols, commas and punctuation marks from a large text, story or paragraph. 
The next step is to convert the tokens into IDs in order to prepare them to get converted into embedding vectos

1. Tokenized using the module re from python the verdict story
2. Convert tokens into IDs. 
-In order to convert tokens into IDs we have to create a vocabulary first. To achieve this, we short the tokens into alphabetical order and map each token into an integer representation called token ID.
3. We have to take vocabular's size in order to use it later to determine model's output space

4. The next step is to create the vocabulary by enumerating the variable which holds all the tokens. After enumerating the tokens we can print some values in order to validate if enumeration succeeded.

#### Professional method
#### The most professional way to do this, is to implement a class called Tokenizer. This class will
#### contain 2 methods, encode && decode. Firstly, I created the vocabulary class containing all the methods described above.
### Step by step
1. Create Tokenizer class
2. Pass in the constructor 2 variables. The first converts the string into int using the vocabulary
-The second converts the integer into string 
3. Then I created the encode method which accepts a raw text, splits the text into words symbols and punctuation marks. Then this method using the variable which has been delegated with the vocabulary creates the vocabulary which contains tokens and integers
4. Then I create the decode method which accepts the returned ids from the vocabulary and converts them back to text

### The process till now is the following: By creating the vocabulary the algorithm has a dictionary to match workds with IDs. When a new text is given, the program digs up this dictionary and by the IDs it find the correct words.
5. However, when there are unknown words, the program crashes. That's why we have to append into the dictionary the "<|unk|>". This compared with an additional line of code in encoder `preprocessed = [item if item in self.str_to_int else <|unk|> for item in preprocessed]`, will return unknown if it cannot match the word in the dictionary. Additinally, I appended in the dict the " <|endoftext|> ". This is being added between different texts to aknwoledge the algorithm that the previous text is unrelated to the next one.

#### Other Useful token words to use:
 [BOS] (beginning of sequence)—This token marks the start of a text. It signifies to
the LLM where a piece of content begins.
 [EOS] (end of sequence)—This token is positioned at the end of a text and
is especially useful when concatenating multiple unrelated texts, similar to
<|endoftext|>. For instance, when combining two different Wikipedia articles
or books, the [EOS] token indicates where one ends and the next begins.
 [PAD] (padding)—When training LLMs with batch sizes larger than one, the
batch might contain texts of varying lengths. To ensure all texts have the same
length, the shorter texts are extended or “padded” using the [PAD] token, up to
the length of the longest text in the batch.


### The steps mentioned above described how tokenization works under the hood. However, real pros use tiktoken which implementes the Byte Pair Encoder (BPE) used in GPT models. I am going to implement it using tiktoken. Tiktoken already contains the vocabulary so you skip the vocabulary creation!

# Installation
1. pip install tiktoken
2. Initialize tokenizer getting gpts encoding `self.tokenizer = tk.get_encoding("o200k_base")`
"o200k_base" is the latest. Search for the latest if there's any newer



#### Another import step is the window slider to teach the model to predict the next token (word).
### This step generate the input–target pairs required for training an LLM
## Explanation
### The sliding window is the step to create the dataset in order to train the GPT like model.
## The flow goes like:
## raw_text -> tokenization -> Sliding window -> Input/Target tensors ->Dataset ->Dataloader
## It is called window slider because the code splits the data in 2 tensors. The first tensor contains the initial data, while the second the predicted token. To clarify, consider the list contains index 0, index 1, etc. Each index represent a token (word) from a given text. The second sensor contains the next token which is in the same index as the previous word in the first tensor. e.g. tensor 1 ['i','am','george'] tensor2['am','george']. The index 0 in the 1st tensor is 'i' while in the tensor2 is 'am'. The input is 'i' the output of the model is 'm'
## Implementaion
1. Tokenization
- Use the tokenization class methods in order to endode the given text: Text ->[t0, t1, t2, t3, t4, t5, ...]
2. Sliding window
- We use max_length to define each list's length (how many token it will contain) and stride to define the next index in the target list: e.g:
max_length = 4
stride = 2
Input 1 → [t0, t1, t2, t3] 
Input 2 → [t2, t3, t4, t5] 
Input 3 → [t4, t5, t6, t7]
Each list contains 4 tokens and each next list starts 2 indexes right from the previous one:
Input 1 → [t0, t1, t2, t3] 
Input 2 → [t2, t3, t4, t5] 
Input 2 starts from t2, which is 2 indexes right from t0
# Tip
- To avoid overlapping use as stride's value, max_length's value
Example:
max_length = 4
stride = 4
Input 1 → [t0, t1, t2, t3] 
Input 2 → [t4, t5, t6, t7] 
Input 3 → [t8, t9, t10, t11]
Each list now does not contains values from the previous one

3. Input και Target
For each input sequence create a target sequence
Input → [t0, t1, t2, t3] 
Target → [t1, t2, t3, t4]
That's how the model learns to predict the next word

4. Convert to pytorch sensors
Each input_token_id is being converted into pytorch sensor and gets saved into the dataset
self.input_ids 
self.target_ids
GPTDatasetV1 class is responsible to do the above described task.
* __len__() → returns the available training examples.
*__getitem__(idx) → returns a specific (input, target) pair.


5. DataLoader και Batch Size
Dataloader from pytorch takes the dataset and groupify them into batches. Batches is being defined by the user
Example:


max_length = 4
batch_size = 2

each batch can be:

Batch:
    [t0, t1, t2, t3]
    [t2, t3, t4, t5]

max_length → how many tokens each sequence has
stride → how much sliding window moves
batch_size → how many sequences each batch contains

# Tip 
Use drop_last=True to drop the last incomplete batch if exists.


## Create token embeddings
The last step in preparing the input text for LLM training is to convert the token IDs
into embedding vectors. As a preliminary step, we must initialize these embedding weights with random values.
This initialization serves as the starting
point for the LLM’s learning process.
The embedding weights are initialized with random values and are learned during training. 
The embedding matrix contains one vector for each token in the vocabulary. 
The token ID is used as an index to look up the corresponding embedding vector.
However, this token embedding does not contain information about the token’s position in the sequence. 
Since self-attention is not inherently aware of token order, positional information is added. 
Absolute positional information tells the model the exact position of a token in the sequence, while relative positional information describes the position of a token relative to other tokens.
## Example
The embedding matrix contains one vector for each token in the vocabulary, and the token ID is used to look up its corresponding embedding vector.
However, the token embedding itself does not contain information about the token’s position in the sequence. 
For example, in the sentence “The bank is near the river,” the model needs to know not only what the token bank is, but also where it appears and how it relates to other tokens such as near and river.
This contextual information helps the model understand that bank refers to a river bank rather than a financial institution.
gAbsolute positional information tells the model the exact position of a token, while relative positional information describes how tokens are positioned in relation to one another.

#Step by step
1. Create the original embeddings using the data loader
2. Look up the token embeddings
3. Create a new layer of embeddings representing the positional embeddings
4. Create the positions using torch.arrange
5. Look up the positional embeddings
6. Combine token and positional embeddings



## Self attention mechanish
The next step is to implement the self-attention mechanism. This mechanism allows the model to read a whole text in one shot in order not to forget the previous word
There are a lot of variants of self-attentions mechanisms. The self-attention simple, the self-attention, the casual attention and the multi-head attention mechanism
The self attention mechanism allows the model to to get selective access into different parts on the input. The attention weights provide an importance in each element of the sequence.
The self-attention allows in each position of a sequence to watch all the other seats of the same sequence. For each token in the sequence a context vector is being computed, an enriched embedding which compines
information from all the tokens in the input, weighted by the attention weights.It is a key component of the transformer architecture and of GPT models.

## Description of self attention and scores
In self-attention, the input text is first converted into a sequence of token embeddings, where each token is represented by a vector (for example, a 3-dimensional vector per word). 
Then, each token is used in turn as the query and compared with every token in the sequence, including itself. 
Each comparison is a dot product: the elements at the same position in the two vectors are multiplied, and the products are summed into a single number, called the attention score (ω). 
For a sentence of T tokens, this gives T × T scores, one for every pair of tokens. Each score reflects how relevant one token is to another. Afterwards a new matrix, with attention scores is being created in order to create the context vector.
 The scores are only the first intermediate step: they are later turned into attention weights, which are used to combine all input vectors into a context vector for each token, an enriched embedding that contains information from the whole sequence. This is usefull in order to understand the relation between words in a sentence.

## Notice
1. A higher attention score means the input token is more relevant to the query token.

# Steps
1. Create the embeddings with their position summed up
2. Calculate the scores by finding the dot product of the embeddings
### 
To compute fast the scores we can use matrix multiplication
attention_scores = inputs @ inputs.T
Actually, we multiply the embeddings matrix with it's transpose
###
3. Normalize the scores using PyTorch.softmax() function
####
Normalizing the scores so that the weights sum to 1 makes the context vector a weighted average of the input vectors. This keeps its scale comparable to the inputs, keeps training stable, and lets each weight be read as the share of attention given to a token. In a few words, normalization normalize the sum of the weights into 1 in order to help the model realise the percentage of attention each token has. In the word journey, if the weight is 0.24 with sum 1, the model understands that it has the 24% of attention. Since the sum is stable, when an element takes more attention the other take less and the model figures out where to give more attention.
####
4. Calculate the context vector by multiplying the
embedded input tokens, x(i), with the corresponding attention weights and then summing
the resulting vectors. A context vector is a list of numbers that captures the meaning of text based on its surrounding words


## Attention Weights
Trainable self-attention: each input embedding x is projected by three trainable weight matrices (W_q, W_k, W_v) into a query, a key and a value vector via matrix multiplication. 
Here d_in = 3 and d_out = 2, so each projected vector has 2 elements. Keys and values are computed for all tokens, since they are all involved in computing the attention weights for the query. Note: "weight parameters" (the W matrices, learned during training) are not the same as "attention weights" (dynamic, context-specific values).
The trainable weight matrices W_q, W_k and W_v are initialized with random values (fixed by a seed for reproducibility) and are optimized during training.

In self-attention, we transform the input vectors in the input matrix X with the three weight
matrices, Wq, Wk, and Wv. The new compute the attention weight matrix based on the resulting queries (Q) and
keys (K). Using the attention weights and values (V), we then compute the context vectors (Z).

# Steps to compute the context vector with trainable weigths
1. Take the input embeddings
2. Initialize the trainable weigths using nn.Linear to create W_Query/W_key/W_value with random values. The same three matrices are shared by all tokens.
3. Calculate the dot product by multiplying inputs seperately with W_Query/W_key/W_value
4. Calculate attention scores by multiplying the value of input @ W_key.T (transpose) with value of input @ W_key
5. Normalize scores using softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) to create the attention weigths
6. Calculate the context vector by multiplying attention weigths @ value of input @W_value

# Tips
The reason for the normalization by the embedding dimension size is to improve the
training performance by avoiding small gradients. For instance, when scaling up the
embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large
dot products can result in very small gradients during backpropagation due to the
softmax function applied to them. 
As dot products increase, the softmax function
behaves more like a step function, resulting in gradients nearing zero. 
These small gradients can drastically slow down learning or cause training to stagnate.
The scaling by the square root of the embedding dimension is the reason why this
self-attention mechanism is also called scaled-dot product attention.


## Why query, key, and value?
The terms “key,” “query,” and “value” in the context of attention mechanisms are
borrowed from the domain of information retrieval and databases, where similar concepts
are used to store, search, and retrieve information.
A query is analogous to a search query in a database. It represents the current item
(e.g., a word or token in a sentence) the model focuses on or tries to understand.
The query is used to probe the other parts of the input sequence to determine how
much attention to pay to them.
The key is like a database key used for indexing and searching. In the attention mechanism,
each item in the input sequence (e.g., each word in a sentence) has an associated
key. These keys are used to match the query.
The value in this context is similar to the value in a key-value pair in a database. It
represents the actual content or representation of the input items. Once the model
determines which keys (and thus which parts of the input) are most
##



## Casual attention
## Casual attention is an attention mechanism which hides the future words in order for the model to consider only the previous words which appear before the current position. It restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores
## To achieve this we code an attention mechanism which masks the tokens after the current input normalize the nonmasked attention weights such that the attention weights sum to 1 in each row. This is essential since the model will get frustrated by having access the future words.
## Example
input 1 : Your [1.0] masked[2.0] masked[3.0] 
input 2: Journey Your [1.0] [2.0] masked[3.0]
input 3: Starts Your [1.0] [2.0] [3.0]
## Tip
One way to obtain the masked attention weight matrix in causal attention is to apply the
softmax function to the attention scores, zeroing out the elements above the diagonal and normalizing
the resulting matrix.

## Tip
Information leakage
When we apply a mask and then renormalize the attention weights, it might initially
appear that information from future tokens (which we intend to mask) could still influence
the current token because their values are part of the softmax calculation. 

However, the key insight is that when we renormalize the attention weights after masking,
what we’re essentially doing is recalculating the softmax over a smaller subset (since masked positions don’t contribute to the softmax value).

The mathematical elegance of softmax is that despite initially including all positions
in the denominator, after masking and renormalizing, the effect of the masked positions
is nullified—they don’t contribute to the softmax score in any meaningful way.

In simpler terms, after masking and renormalization, the distribution of attention
weights is as if it was calculated only among the unmasked positions to begin with.
This ensures there’s no information leakage from future (or otherwise masked)
tokens as we intended.


# Tip
A more efficient way to obtain the masked attention weight matrix in
causal attention is to mask the attention scores with negative infinity values before
applying the softmax function.

## Tip
In order to avoid overfitting through training a deep learning method is Dropout.
By using dropout the trainer drops some random elements to zero, in order for the model not to get a biased training. This method helps prevent overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units. It’s important to emphasize that dropout is only used
during training and is disabled afterward.

## Steps
1. Update SelfAttention class to casual attention and add a buffer
buffers are automatically moved to the appropriate device (CPU or GPU) along with our model, which will
be relevant when training our LLM. This means we don’t need to manually ensure
these tensors are on the same device as your model parameters, avoiding device mismatch errors.
2. Update forward method in order to implement the casual attention with a mask

## Multihead attention mechanism
In practical terms, implementing multi-head attention involves creating multiple
instances of the self-attention mechanism, each with its own weights,
and then combining their outputs. 
Using multiple instances of the self-attention
mechanism can be computationally intensive, but it’s crucial for the kind of complex
pattern recognition that models like transformer-based LLMs are known for.
The main idea is to run the attention mechanism multiple times in paraller with different linear projects - the results of multiplying  the input data (like the key, query and values) by a weight matrix. 
In code, we can achieve this by implementing a simple
MultiHeadAttentionWrapper class that stacks multiple instances of our previously
implemented CausalAttention module.

# Steps
1. Create a multihead wrapper class and create an instance of the casual attention class with nn.Module
2. Create the forward method and concatenate the two context vectors
3. From now on use the multihead wrapper class with an argument num_heads to define how many instances of the
casual attention you want to create

# Another usefull tip is to implement head process in parallel. One way to achieve this is to compute the output of the multihead attention mechanism simultaneously via matrix multiplication

# Implementing multi-head attention with weight splits

#### Summary
 Attention mechanisms transform input elements into enhanced context vector
representations that incorporate information about all inputs.
 A self-attention mechanism computes the context vector representation as a
weighted sum over the inputs.
 In a simplified attention mechanism, the attention weights are computed via
dot products.
 A dot product is a concise way of multiplying two vectors element-wise and then
summing the products.
 Matrix multiplications, while not strictly required, help us implement computations
more efficiently and compactly by replacing nested for loops.
 In self-attention mechanisms used in LLMs, also called scaled-dot product
attention, we include trainable weight matrices to compute intermediate transformations
of the inputs: queries, values, and keys.
 When working with LLMs that read and generate text from left to right, we add
a causal attention mask to prevent the LLM from accessing future tokens.
 In addition to causal attention masks to zero-out attention weights, we can add
a dropout mask to reduce overfitting in LLMs.
 The attention modules in transformer-based LLMs involve multiple instances of
causal attention, which is called multi-head attention.
 We can create a multi-head attention module by stacking multiple instances of
causal attention modules.
 A more efficient way of creating multi-head attention modules involves batched
matrix multiplications.


## Multi-Head Attention with Weight Splits — Summary

* **Self-attention** transforms each input token into a richer **context vector** by incorporating information from other tokens in the sequence.

* A context vector is computed as a **weighted sum of the input representations**, where the weights determine how much attention each token receives.

* In self-attention, **attention weights** can be obtained from the similarity between **queries and keys**, using dot products.

* A **dot product** multiplies corresponding elements of two vectors and then sums the results, providing a measure of their similarity.

* **Matrix multiplication** allows these operations to be performed efficiently in parallel, replacing explicit nested loops with compact tensor operations.

* In LLMs, self-attention is implemented as **scaled dot-product attention**, where trainable weight matrices transform the input embeddings into:

  * **Queries (Q)** — what each token is looking for.
  * **Keys (K)** — what each token offers for matching.
  * **Values (V)** — the information that is aggregated.

* For autoregressive language models, a **causal attention mask** prevents each token from attending to future tokens, ensuring that predictions only depend on previous and current tokens.

* **Dropout** can be applied to the attention weights during training to reduce overfitting and improve generalization.

* **Multi-head attention** combines multiple attention heads, allowing the model to learn different types of relationships between tokens simultaneously.

* A straightforward implementation creates multiple independent attention modules and concatenates their outputs.

* A more efficient implementation performs the computations for all attention heads **in parallel using batched matrix multiplications**, typically by splitting the projected Q, K, and V representations into multiple heads.

### Key Idea

Instead of processing each attention head separately:

```text
Head 1 → Attention
Head 2 → Attention
Head 3 → Attention
...
```

we organize the tensors so that all heads can be processed simultaneously:

```text
Input Embeddings
       ↓
   Q, K, V
       ↓
   Split into heads
       ↓
Batched Matrix Multiplication
       ↓
Attention for all heads
       ↓
Concatenate heads
       ↓
Output Projection
       ↓
Context Representations
```

The main advantage is **computational efficiency**: the mathematical operation remains the same, but modern hardware can process the attention heads in parallel.




## Transformer blocks and layer normalization
The GPTModel class defines the full architecture of a GPT-style language model: it converts input token IDs into token embeddings via tok_emb, adds positional embeddings via pos_emb so the model knows the order of tokens, applies dropout for regularization, and passes the result through a stack of transformer blocks (trf_blocks)
After the transformer blocks, a Normalization layer is applied to stabilize the activations by rescaling them to have zero mean and unit variance (using two learnable parameters, scale and shift, so the network can adjust this normalization during training), before a final linear layer (out_head) projects the normalized embeddings into logits — one raw, unnormalized score per vocabulary token, for each position in the sequence — which represent the model's unnormalized predictions for the next token and would later be converted into probabilities via a softmax function.



### Τhe Architectural Pieces of a Transformer
### When you combine LAYER NORMALIZATION, GELU,  Feed-Forward Network, SHORTCUT CONNECTION and a loss function you are looking at the core machinery of a Transformers Block (GPT AND LLAMA)

Here is a clear, high-level summary of what each component does and why it matters

1. Layer Normalization: It works as the stabilizer. It rescales the embedding values for each individual word in order not to become very
large or very small. 
The maths behind it: It forces the features to have a mean of 0 and variance of 1. 
Importance: It's crucial because it prevents the  neural network from crashing or stalling during training due to extreme values.


2. GELU Activation: The GATEKEEPER: A smooth valve that decides which numbers are important enough to pass them through the next layer.
The maths behind it: It multiplies the input in a propability curve based on the normal distribution.
Importance: Introduces non-linearity [1,1]. Without it a deep network is just a giant linear equation which cannot learn difficult patterns

3. FEED-FORWARD-NETOWRK (FFN): The ENGINE: A small network inside the block which gets a word's vector, expands it to a massive size
and shrinks it to analyze it, and shrinks it back down.
The maths behind it:  Projects Dimensions up (e.g. 758 dimensions multiplied by 4,e.g., 768 → 3072). Applies GELU ACTIVATION, 
and projects them back down (3072->768)
Importance: It's crucial because it gives the A.I the necessary thinking space or memory capacity to combine features and learn complex contepts

4. Shortcut Connection: The HIGHWAY. A bypass lane that allows the original input data to skip a layer and add itself directly to the output.
The maths behind it: Adds the input X into the layer's output: output = X+layer(x)
Importance: Solves the GRADIENT VANISHING PROBLEM. It passes signals directly backward during training

5. Loss Function: The SCOREKEEPER. A mathematical ruler which calculates how far is the AI's current guess from the real target
The maths behind it: Calculates error (like MSE for numerical differences, or Cross-Entropy for word predictions).

The combination of flow data
```
[ Input Tensor x ]
       │
       ├───► (Highway Bypass Lane) ──────────────────────────────────┐
       ▼                                                             │
[ Layer Normalization ]  ──► Stabilizes numbers (Mean=0, Var=1)      │
       ▼                                                             │
[ Feed-Forward Network ] ──► Linear Up ──► [ GELU ] ──► Linear Down  │
       ▼                                                             │
[ Output of Layer ]                                                  │
       ▼                                                             │
[ Shortcut Connection ]  ◄── (Add the original x back in) ◄──────────┘
       ▼
[ Final Prediction ] ────► Compare with Target via [ Loss Function ] ──► Compute Gradients
```

