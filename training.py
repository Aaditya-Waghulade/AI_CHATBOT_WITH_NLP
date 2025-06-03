#We will load the data from intents.json here and train the data
#1. Common Py libs
import json
import random
import pickle
import numpy as np

#2. nlp libs for Training data
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('wordnet')
#3. Tensorflow-keras for layers , activation function for building neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation
from tensorflow.keras.optimizers import SGD

#SGD : Stochastic Gradient Descent is an optimization algorithm used to minimize the loss function and update the model parameters.
#"Stochastic" means we don’t use all data at once. We use just one data point or a small batch at a time. This makes it faster and works well for big datasets.

# A) LOADING DATA

#4.Opening the "intents.json" file
with open("intents.json") as intents_json:
    data = intents_json.read()

#Loading data into json format
intents = json.loads(data)

#Empty lists for storing operations
classes = []
words = [] #For Tokens
documents = []
ignore_letters = ['?',',','.','!']#for storing some letters which are not really important

#5. Looping through each intent in the data and then tokenize its "patterns" 
for intent in intents['intents']: # 1.intents = json.loads(data) 2. intents['intents'] =  'intents.json'==>{intents :[]}
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) #extend() means taking the content in the list and append() means adding the another list in a list
        documents.append((word_list,intent['tag'])) #document =[] will contain every word and class which are intent:tag in intents.jsone
        if  intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)

#LOADING DATA IS DONE

# B) TRAINING THE DATA
#6. Lemmatizing the words
lemmatizer = WordNetLemmatizer()
for word in words:
    if word not in ignore_letters:
        words = [lemmatizer.lemmatize(word)]
#Lemmatizing the each word by iterating the words list because it is now sorted. then entering in list words and the word is not matching with ignoring letters then add that word in words[] by using lemmatizer
print(words)