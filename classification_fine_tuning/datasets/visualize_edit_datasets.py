
#this contains code which I use to visualize the datasets
import pandas as pd

import os
from pathlib import Path

data_file_path = 'C:/Users/Overkill/Desktop/train-llm/BaseLLM/classification-fine-tuning/sms_spam_collection/SMSSpamCollection.tsv'
#visualize sms dataset
df = pd.read_csv(
data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
#print(df)

#examine label distributions
#print(df["Label"].value_counts())

#this method drops out ham rates in order to match with spam rates, since we want to imbalance the dataset
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] #count spam instances
    ham_subset = df[df["Label"] == "ham"].sample(
    num_spam, random_state=123 #drop ham instances to match spam instances
    )
    balanced_df = pd.concat([                #compine ham subset with spam
    ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())


# Now let's convert the string into integer class numbers 0 and 1 respectively
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

"""Split the dataset into 3 parts, 70% training, 10% validation sets and 20% testing, which is a common method during machine learning"""
def random_split(df, train_frac, validation_frac):
    df = df.sample(
    frac=1, random_state=123
    ).reset_index(drop=True) # shuffle the entire dataframe
    train_end = int(len(df) * train_frac) #calculate split indices
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end] #split the dataframe 
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


#execute the data split method
train_df, validation_df, test_df = random_split(
balanced_df, 0.7, 0.1) #since 0.7 is the training data, 0.1 the validation data, the 0.2 remaing is the testing data

#save into a csv file the dataset
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)