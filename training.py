#We will load the data from intents.json here and train the data
#1. Common Py libs
import json
import random
import pickle
import numpy as np

#2. nlp libs for Training data
import nltk
from nltk.stem import WordNetLemmatizer

#3. Tensorflow-keras for layers , activation function for building neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation
from tensorflow.keras.optimizers import SGD
#SGD : Stochastic Gradient Descent is an optimization algorithm used to minimize the loss function and update the model parameters.
#"Stochastic" means we don’t use all data at once. We use just one data point or a small batch at a time. This makes it faster and works well for big datasets.

lemmatizer = WordNetLemmatizer()

#4.Opening the "intents.json" file
with open("intents.json") as intents_json:
    data = intents_json.read()

#Loading data into json format
intents = json.loads(data)

#Empty lists for storing operations
classes = []
words = [] #For Tokens
documents = []
ignore_letters ['?',',','.','!']#for storing some letters which are not really important

#5. Looping through each intent in the data and then tokenize its "patterns" 
for intent in intents['intents']: # 1.intents = json.loads(data) 2. intents['intents'] =  'intents.json'==>{intents :[]}
    for pattern in intent["patterns"]:
        word_list = nltk.tokenize(pattern)
        words.append(word_list)
        documents.append((word_list),intent['tag']) #document =[] will contain every word and class which are intent:tag in intents,jsone