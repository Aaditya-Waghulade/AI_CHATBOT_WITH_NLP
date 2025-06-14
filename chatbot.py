import json 
import pickle
import numpy as np
import random
import nltk
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
 

#2. NOW CREATING SENTENCE INTO BAG OF WORDS
def bag_of_words(sentence):
    sentence_words = tokenizing_input(sentence)
    bag = [0]*len(words) #Creating a list of zeros with the same length as the words lists

    for w in sentence_words:
        for i, word in enumerate(words): #means for each word in the words list
            if word == w:
                #assign 1 to the index of the word
                bag[i] = 1
    return np.array(bag)


#3. PREDICTING THE INTENT OF THE USER
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array(np.array([bow])))[0] #predicting the output of the model which will be on 0th index
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD ] #getting the index and the value (Probability) of the result
    results.sort(key=lambda x:x[1], reverse=True) #reversing the list so that the highest probability is at the top
    #x[1] because we want to sort based on the second element of the list from the results list

    return_list = []
    for r in results:
        return_list.append({
            'intent': classes[r[0]],
            'probability':str(r[1])
        })
    return return_list
#Why do we use enumerate(res) here?
'''We want to:
Identify which classes have a probability greater than the threshold (ERROR_THRESHOLD = 0.25).

Keep both the class index and its probability, so we can later map it to a class name.'''


#4. Getting response
def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses']) #responses from intents.json
            break
    return result

while(True):
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints,intents)
    print("JARVIS: ", res)

