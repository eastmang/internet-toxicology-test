from flask import Flask, render_template, request, session
from cleaning.cleaning_functions import clean_string
import os
import random
import numpy as np
import json
from keras.models import load_model
import pandas as pd

os.chdir("D:\Grad 2nd year\Winter Quarter\Data Use\Final Project") # sets to the local directory where the maser folder is


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # This prevents GPU usage only called becaues of an issue on my laptop



model_production = load_model('model_production.h5') # loads the production model
model_beta = load_model('model_beta.h5') # loads the beta model


def get_outcome(correct, toxicity): # This takes in the model prediction and the user input to determine the correct outcome
    if correct is 'yes':
        return toxicity
    else:
        return abs(toxicity-1)  # This turns a 0 to a 1 and a 1 to a 0


def tell_toxic(value): # This makes a string based on the prediction to then be fed into the next webpage
    if value == 1:
        return "toxic"
    else:
        return "not toxic"


app = Flask(__name__)
app.secret_key = str(random.random())

@app.route('/')
def beta():
    return render_template('demo.html') # shows the demo


@app.route('/comment/', methods=['POST', 'GET'])
def comment(): # This takes in the comment that the user wants to see
    if request.method == 'GET':
        return f"The URL /data should not be referenced directly." # ensures that the user doesnt just skip to this url
    if request.method == 'POST':
        session['mod'] = request.form['prod_beta']
    return render_template('input.html')


@app.route('/comment/data/', methods=['POST', 'GET'])
def data(): # This uses the string from the last function and then it cleans the string and makes a prediction
    if request.method == 'GET':
        return f"The URL /data should not be referenced directly."
    if request.method == 'POST':
        phrase = request.form['phrase'] # This takes the user input from the last page
        session['text'] = phrase # This saves the user string in the session
        cleaned = clean_string(phrase) # This cleans the string using a function int he preprosessing file
        session['cleaned'] = json.dumps(cleaned.tolist()) # This saves the cleaned output as a json, necessary because of how the session saves data
        if session.get('mod') == 'beta':
            prediction = round(float(model_beta.predict(cleaned))) # This gets the beta model prediction
        else:
            prediction = round(float(model_production.predict(cleaned))) # This gets the production model prediction
        toxic = tell_toxic(prediction) # This gets the string to give to the user depending on the model output
        session['prediction'] = json.dumps(prediction) # This then saves the prediciton in the session
    return render_template('output.html', phrase=toxic)


@app.route('/comment/data/thanks/', methods=['POST', 'GET'])
def thanks():
    if request.method == 'GET':
        return f"The URL /data should not be referenced directly."
    if request.method == 'POST':
        yes_no = request.form['yes_no'] # This gets the user feedback on the model's answer
        prediction = json.loads(session.get('prediction', None)) # This loads in the session prediciton
        x_val = np.array(json.loads(session.get('cleaned', None))) # This loads in the cleaned string
        y_val = np.array([get_outcome(yes_no, prediction)]) # This makes the correct y output
        model_beta.fit(x_val, y_val) # This updatest he beta model according to the correct answer
        df = pd.DataFrame.from_dict({'toxic': y_val, 'text': session['text']}) # this makes a row for the csv
        df.to_csv('toxic.csv', mode='a', header=False) # This appends the row to the csv
        model_beta.save("model_beta.h5", overwrite=True) # This saves the beta model now that it has been updated
    return render_template('thanks.html')


app.run()
