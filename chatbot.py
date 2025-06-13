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



