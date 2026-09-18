##import tokenizer
from tokenizer_bpe.__tokenizer import tokenizer

#main class
def main():

   #text example
   text = (
           "Akwirw ier"
           )
   #call tokenizer method
   text_to_id = tokenizer.encoder(text)

   ##print all id's
   for i in text_to_id:
      print(i)
   for token in text_to_id:
      print(tokenizer.decoder([token]))

   #call the decoder method
   id_to_string = tokenizer.decoder(text_to_id)

 

   #print results
   #print(text_to_id)
   print(id_to_string)

#execute main
main()
   