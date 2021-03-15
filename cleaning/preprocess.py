import os
import pickle
import pandas as pd
from parameters.hyperparams import RAND
from cleaning.cleaning_functions import clean_frame







os.chdir("D:\Grad 2nd year\Winter Quarter\Data Use\Final Project") # make it the local directory
df = pd.read_csv('toxic.csv') # takes in the dataset
df.rename(columns={"comment_text": "text"}, inplace=True)
df = df [['toxic', 'text']]

# The classes are not balanced so the below code will help to downsample the smaller of the two classes
df_minority = df.loc[df['toxic'] == 1]
df_majority = df.loc[df['toxic'] == 0]
df_majority_downsampled = df_majority.sample(n = len(df_minority), replace = True, random_state = RAND)

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# removing all rows where there is a non-string in the text (there is only 1)

df_downsampled = df_downsampled[df_downsampled['text'].map(type) == str]


# A function to replace contractions that come of from the tokenization
comments = clean_frame(df_downsampled)
df_downsampled.to_csv('toxic.csv')
with open('comments', 'wb') as f: pickle.dump(comments, f)
