import re
import string
import pandas as pd
import nltk
from parameters.hyperparams import MAX_LENGTH, VOCAB_SIZE
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')




def decontracted(phrase): # This function removes contractions and converts them into the correct words
    # specific
    # general
    phrase = re.sub(r"n\'t", "not", phrase)
    phrase = re.sub(r"\'re", "are", phrase)
    phrase = re.sub(r"\'s", "is", phrase)
    phrase = re.sub(r"\'d", "would", phrase)
    phrase = re.sub(r"\'ll", "will", phrase)
    phrase = re.sub(r"\'t", "not", phrase)
    phrase = re.sub(r"\'ve", "have", phrase)
    phrase = re.sub(r"\'m", "am", phrase)
    return phrase


def clean_frame(df): # This cleans the data input
    assert type(df) == pd.DataFrame # makes sure that it is taking in a dataframe
    sentence = [] # we will put the string into this list
    for i in range(len(df)): # go down the dataframe
        tokens = nltk.word_tokenize(df.iat[i,1]) # go through each word in the string and make it a token
        tokens = [decontracted(word) for word in tokens] # replace contractions from the tokens
        cleaned_tokens = ""
        for token, tag in pos_tag(tokens):
            # remove punctuation from each token
            table = str.maketrans('', '', string.punctuation)
            tokens = [word.translate(table) for word in tokens]
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            # filter out short tokens
            tokens = [word for word in tokens if len(word) > 1]
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower():
                cleaned_tokens= cleaned_tokens + " " + token.lower()
        sentence.append(cleaned_tokens)
    # integer encode the documents
    encoded_docs = [one_hot(word, VOCAB_SIZE) for word in sentence]
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')
    return padded_docs


def clean_string(phrase): # This is like the function above bu it cleans only a specific string. Its for the beta model
    sentence = []
    tokens = nltk.word_tokenize(phrase)
    # replace contractions from the tokens
    tokens = [decontracted(word) for word in tokens]
    cleaned_tokens = ""
    for token, tag in pos_tag(tokens):
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower():
            cleaned_tokens= cleaned_tokens + " " + token.lower()
    sentence.append(cleaned_tokens)
    # integer encode the documents
    encoded_docs = [one_hot(word, VOCAB_SIZE) for word in sentence]
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')
    return padded_docs