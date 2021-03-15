#############################################################
#################### IMPORTING LIBRARIES ####################
#############################################################
import keras
import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense,  Input, LSTM, Bidirectional,  Concatenate
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import pickle
from parameters.hyperparams import VOCAB_SIZE, MAX_LENGTH, EPO, BATCHES, VERBOSITY, TESTING_SIZE, RAND
from parameters.hyperparams import LEARNING_RATE, hidden_dim, nodes_lstm

#############################################################
#################### LOADING DATASETS #######################
#############################################################
os.chdir("D:\Grad 2nd year\Winter Quarter\Data Use\Final Project")
comments = pickle.load(open("comments", "rb"))
toxic = pd.read_csv('toxic.csv')

x_train, x_test, y_train, y_test = train_test_split(comments, toxic['toxic'], random_state = RAND, test_size = TESTING_SIZE)


#############################################################
#################### MAKING MODELS ##########################
#############################################################
def make_encoder_decoder():
    input = Input(shape = (MAX_LENGTH))
    x = Embedding(input_dim=VOCAB_SIZE+1, output_dim=hidden_dim, input_length=MAX_LENGTH)(input)
    x = Bidirectional(LSTM(nodes_lstm, return_sequences = True))(x)
    x = Bidirectional(LSTM(nodes_lstm))(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = input, outputs = x)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model



model_encoder_decoder = make_encoder_decoder()


history = model_encoder_decoder.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)


#############################################################
################### GETTING RESULTS #########################
#############################################################
y_pred_encoder_decoder = model_encoder_decoder.predict(x_test).round().ravel()
precision_encoder_decoder = precision_score(np.asarray(y_test), y_pred_encoder_decoder)
accuracy_encoder_decoder = accuracy_score(np.asarray(y_test), y_pred_encoder_decoder)
recall_encoder_decoder = recall_score(np.asarray(y_test), y_pred_encoder_decoder)


print("Encoder_Decoder:", accuracy_encoder_decoder, precision_encoder_decoder, recall_encoder_decoder)

model_encoder_decoder.save('model_production.h5')
