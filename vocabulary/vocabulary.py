####this file contains the data to prepair the tokenizer (encoder + decoder)
#import the verdict text
import re
from pathlib import Path
class Vocabulary:
    def __init__(self, file_path: str):
        self.file_path = Path(__file__).parent / file_path

        self.raw_text = self.import_file_path()
        self.preprocessed = self.tokenize_text(self.raw_text)
        self.sorted_text = self.sort_tokenized_alphabetically()
        self.vocabulary = self.create_vocabulary()


    #import file path method
    def import_file_path(self)-> str:
        ##open the file
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read() ## read the raw text from the file
            return raw_text

        

    #tokenize text method
    def tokenize_text(self,raw_text:str)->list:
        ###try to tokenize the text, seperate all the words and symbols
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        #use strip() to remove the the whitespaces
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #print(len(preprocessed))##print tokens number
        #validate how the words have been split
        return preprocessed
        

   
    #sort the teokenized text alphabetically and get vocabulary's size
    def sort_tokenized_alphabetically(self) -> list:
        sorted_text = sorted(set(self.preprocessed))

         ##this line of code adds the token endoftext 
         # to disclose to the algorithm that the next text is unrelated to the previous one.
         #the token unk represents the unknown token
        sorted_text.extend(["<|endoftext|>", "<|unk|>"])
        #get vocabulary's size
        #vocabulary_size = len(sorted_text)
       # print(vocabulary_size)
        return sorted_text
        




    #create the vocabulary
    def create_vocabulary(self)-> dict:
        ##define a variable and enumerate sorted_text with integers
        vocabulary ={token:integer for integer,token in enumerate(self.sorted_text)}
        
        return vocabulary

