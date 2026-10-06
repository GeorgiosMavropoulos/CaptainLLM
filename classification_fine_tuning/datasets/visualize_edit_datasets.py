
#this contains code which I use to visualize the datasets
import pandas as pd

import os
from pathlib import Path
"""
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

#Split the dataset into 3 parts, 70% training, 10% validation sets and 20% testing, which is a common method during machine learning
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
  """


DATASETS_DIR = Path(__file__).resolve().parent

FILES = {
    "train": DATASETS_DIR / "train_new.csv",
    "validation": DATASETS_DIR / "val_new.csv",
    "test": DATASETS_DIR / "test_new.csv",
}


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

datasets = {}

for split, path in FILES.items():
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{path.name} πρέπει να έχει columns: text, label"
        )

    # Convert text to string and remove surrounding whitespace
    df["text"] = df["text"].astype(str).str.strip()

    datasets[split] = df

    print(f"\n{'=' * 60}")
    print(f"{split.upper()}")
    print(f"{'=' * 60}")
    print(f"Rows: {len(df)}")


# --------------------------------------------------
# 1. Class balance
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("CLASS BALANCE")
print(f"{'=' * 60}")

for split, df in datasets.items():

    counts = df["label"].value_counts()

    print(f"\n{split.upper()}:")
    print(f"Total: {len(df)}")

    for label, count in counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")


# --------------------------------------------------
# 2. Duplicates inside each split
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("DUPLICATES INSIDE EACH SPLIT")
print(f"{'=' * 60}")

for split, df in datasets.items():

    duplicates = df[df.duplicated(subset=["text"], keep=False)]

    print(f"\n{split.upper()}:")
    print(f"Duplicate rows: {len(duplicates)}")

    if len(duplicates) > 0:
        print(duplicates.sort_values("text").to_string(index=False))


# --------------------------------------------------
# 3. Duplicate texts between splits
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("DUPLICATES BETWEEN SPLITS")
print(f"{'=' * 60}")

split_names = list(datasets.keys())

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):

        split_a = split_names[i]
        split_b = split_names[j]

        df_a = datasets[split_a]
        df_b = datasets[split_b]

        texts_a = set(df_a["text"])
        texts_b = set(df_b["text"])

        duplicates = texts_a.intersection(texts_b)

        print(
            f"\n{split_a.upper()} <-> {split_b.upper()}: "
            f"{len(duplicates)} duplicated texts"
        )

        if duplicates:
            for text in duplicates:
                print(f"  {text[:150]}")


# --------------------------------------------------
# 4. Same text with different labels
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("SAME TEXT WITH DIFFERENT LABEL")
print(f"{'=' * 60}")

all_data = pd.concat(
    datasets.values(),
    ignore_index=True
)

label_counts = (
    all_data
    .groupby("text")["label"]
    .nunique()
)

conflicting_texts = label_counts[label_counts > 1]

print(f"\nConflicting texts: {len(conflicting_texts)}")

if len(conflicting_texts) > 0:

    for text in conflicting_texts.index:

        rows = all_data[all_data["text"] == text]

        print("\nTEXT:")
        print(text)

        print("LABELS:")
        print(rows[["label"]].drop_duplicates().to_string(index=False))


# --------------------------------------------------
# 5. Overall dataset statistics
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("OVERALL DATASET")
print(f"{'=' * 60}")

print(f"Total rows: {len(all_data)}")
print(f"Unique texts: {all_data['text'].nunique()}")

total_duplicates = len(all_data) - all_data["text"].nunique()

print(f"Duplicate texts overall: {total_duplicates}")


# --------------------------------------------------
# 6. Final summary
# --------------------------------------------------

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print(f"{'=' * 60}")

print("\nSplits:")
for split, df in datasets.items():
    print(f"  {split}: {len(df)} rows")

print(f"\nOverall duplicate texts: {total_duplicates}")
print(f"Conflicting labels: {len(conflicting_texts)}")

if total_duplicates == 0:
    print("✓ No duplicate texts found.")

if len(conflicting_texts) == 0:
    print("✓ No texts with different labels found.")

print("\nDone.")