###tokenize the verdict text
import re #import re module to split the text
#import the vocabulary class
from vocabulary.vocabulary import Vocabulary

#create a vocabulary instance

class Tokenizer:
    def __init__(self,file_path: str):
        vocab = Vocabulary(file_path) ###when the class is being initialized it automatically import the given file
        ### use a variable string to int and assign the create vocabulary method
        self.string_to_int =  vocab.vocabulary
        #create a variable int to string in order to convert the text back to string
        #this inverts the dictionary from 'home':123 to 123:'home'
        self.int_to_string = {i:s for s,i in self.string_to_int .items()}


    #create the encoder method
    def encode(self,text:str) -> list[int]:
        ##split the text word by word, symbols and punctuation marks
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        #clean the whitespaces
        preprocessed =[item.strip() for item in preprocessed if item.strip()]
        ##this line of code forces the algorithm to return "<|unk|>" for unknwon words
        preprocessed = [item if item in self.string_to_int
                                else "<|unk|>" for item in preprocessed]
        #map each token with its id
        ids = [self.string_to_int[s] for s in preprocessed]
        return ids

    #create the decoder
    def decode(self,ids: list[int]) ->str:
        text = " ".join([self.int_to_string[i] for i in ids])#iterate the ids list and revert the order -> token:id

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)### remove the whitespaces

        return text







      


