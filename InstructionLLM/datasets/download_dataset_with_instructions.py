"""This file contains the required code to download the required dataset with instructors"""
import json
import os
import urllib
from urllib import request


def download_and_load_file(file_path, url): #download the required dataset method
 if not os.path.exists(file_path):
  with request.urlopen(url) as response:
   text_data = response.read().decode("utf-8")
  with open(file_path, "w", encoding="utf-8") as file:
    file.write(text_data)
    #if the dataset has been already downloaded before skip the process
 else: 
    with open(file_path, "r", encoding="utf-8") as file:
     text_data = file.read()
 with open(file_path, "r") as file:
    data = json.load(file)
 return data


file_path = "instruction-data.json"
url = (
"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
"/main/ch07/01_main-chapter-code/instruction-data.json"
)

#download the file
data = download_and_load_file(file_path,url)
#print data length
print(f"Data length:{len(data)}")

#print a line
print(f"Data example:\n{data[10]}")