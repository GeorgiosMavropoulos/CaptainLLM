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






