import json 
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer


lemmatizer=WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

#FLOW OF THE PROGRAM
#1. TOKENNIZING AND LEMMATIZING USER INPUT
def tokenizing_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words ]
    return sentence_words
 

#2. NOW CREATING SENTENCE INTO BAG OF WORDS (List of 1 and 0 which will indicates whether that word is Avaialble in the Bag Or Not)