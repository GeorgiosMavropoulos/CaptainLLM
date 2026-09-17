##import tokenizer
from tokenizer.tokenizer import Tokenizer
from vocabulary.vocabulary import Vocabulary



#test to see if it works
text1 = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
text2 = "Hello, do you like tea?"

##connect the strings
text = "<|endoftext|> ".join((text1,text2)) #join the texts and add between the endoftext token
tokenizer = Tokenizer("the_verdict.txt")
ids = tokenizer.encode(text2)
#try to decode it
decoded_text = tokenizer.decode(ids)

print(ids)
print(decoded_text)