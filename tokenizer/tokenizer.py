###tokenize the verdict text
import re #import re module to split the text
#import the verdict text
import os

file_path = "the_verdict.txt"
##open the file
with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() ## read the raw text from the file
    

###try to tokenize the text, seperate all the words and symbols
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
#use strip() to remove the the whitespaces
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))##print tokens number
#validate how the words have been split
print(preprocessed[:30])