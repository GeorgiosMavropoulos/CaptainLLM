##import tokenizer
from tokenizer_bpe.__tokenizer import tokenizer
from dataloader.windowslider import LoadData
#main class
"""
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
   """

##test DataLoad from pytorch
def main():
 #instanciate LoadData 
 load_data_pytorch = LoadData()

 def load_text():
  with open("datasets/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
  return raw_text

 #load text
 initial_text = load_text()

 ##call create_dataset_v1 from LoadData class 
 # in order to prepare the test and train the model using the auto-regressive method.
 #For this example I will use a small batch_size
 dataloader =load_data_pytorch.create_dataloader_v1( initial_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False)
 #use the iter which uses python's next function 
 # in order to iterate through the batches and fetch the next entry
 data_iter = iter(dataloader)
 inputs, targets = next(data_iter)
 print("Inputs:\n", inputs)
 print("\nTargets:\n", targets)


#call the main method
main()
 

 

