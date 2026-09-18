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


### The steps mentioned above indicate how to tokenization works under the hood. However, real pros use tiktoken which implementes the Byte Pair Encoder (BPE) used in GPT models. I am going to implement it using tiktoken. Tiktoken already contains the vocabulary so you skip the vocabulary creation!

# Installation
1. pip install tiktoken
2. Initialize tokenizer getting gpts encoding `self.tokenizer = tk.get_encoding("o200k_base")`
"o200k_base" is the latest. Search for the latest if there's any newer




