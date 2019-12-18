# NOTE: Jay Allamar's tutorial (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) is followed to encode the messages using DistilBERT

# !pip install transformers # If you run this in colab you need to first install the transformers package
import re
import torch
import string
import numpy as np
import pandas as pd
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from google.colab import files
#uploaded = files.upload() # If you run this on colab you need to upload the spam.csv file to colab

# Loading the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Dropping unwanted columns
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)

# Chaning the labels for convinience
df["v1"].replace({"ham": 0, "spam":1}, inplace=True)

# Changing the column names for better 
df.rename({"v1": "spam", "v2": "original_message"},axis=1, inplace=True)


#import nltk
#nltk.download("stopwords")
#nltk.download("punkt") # again if your running this in colab you'll probably need to first download the stopwords set and punkt from nltk

def clean_sentence(s):
    """Given a sentence remove its punctuation and stop words"""
    
    stop_words = set(stopwords.words('english'))
    s = s.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    tokens = word_tokenize(s)
    cleaned_s = [w for w in tokens if w not in stop_words] # removing stop-words
    return " ".join(cleaned_s[:30]) # using the first 30 tokens only

# Clean the sentences
df["cleaned_message"] = df["original_message"].apply(clean_sentence)


# Loading pretrained model/tokenizer
# This is the Distilled, base, uncased version of BERT 
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

# Tokenize the sentences adding the special "[CLS]" and "[SEP]" tokens
tokenized = df["cleaned_message"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Get the length of the longest tokenized sentence
max_len = tokenized.apply(len).max() 

# Padd the rest of the sentence with zeros if the sentence is smaller than the longest sentence
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]) 

# Create the attention mask so BERT knows to ignore the zeros used for padding
attention_mask = np.where(padded != 0, 1, 0)

# Create the input tensors
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

# Pass the inputs through DistilBERT
with torch.no_grad():
    encoder_hidden_state = model(input_ids, attention_mask=attention_mask)

# Create a new dataframe with the encoded features
df_encoded = pd.DataFrame(encoder_hidden_state[0][:,0,:].numpy())

# Insert the original columns in the beginning of the encoded dataframe
df_encoded.insert(loc=0, column='original_message', value=df["original_message"])
df_encoded.insert(loc=0, column='spam', value=df["spam"])

# Download the encoded csv
df_encoded.to_csv("./output/spam_encoded.csv", index=False)