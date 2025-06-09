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

#5. Looping through each intent in the data and then tokenize its "patterns or question" 
for intent in intents['intents']: # 1.intents = json.loads(data) 2. intents['intents'] =  'intents.json'==>{intents :[]}
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) #extend() means taking the content in the list and append() means adding the another list in a list
        documents.append((word_list,intent['tag'])) #document =[] will contain every word and class which are intent:tag in intents.jsone
        if  intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)
#So word_list is the "patterns" or questions from intents

# B) TRAINING THE DATA (PRE_PROCESSING)
#6. Lemmatizing the words list
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#
#Lemmatizing the each word by iterating the words list because it is now sorted. then entering in list words and the word is not matching with ignoring letters then add that word in words[] by using lemmatizer
#now removes duplicates from the words list by using set
words = sorted(set(words))
classes = sorted(set(classes))
#----------------________-NO ERROR TILL HERE___________-------------

#7. creating pickle file for words and classes list 
pickle.dump(words , open('words.pkl','wb'))#Putting words list in words.pkl 
pickle.dump(classes,open('classes.pkl','wb'))#Putting classes list in classes.pkl

'we cant feed these words directly to the neural network so first we have to make them in numerical value '
"We are using BAG OF WORDS FOR THIS"
#8. Creating a bag of words
training = []
output_empty = [0]*len(classes)

#9. Looping through each document in the documents list
#Because we have to make whole document in numerical value
for document in documents: 
    bag = []#Creating a bag for each document which contain the words and their frequency
    word_patterns = document[0] # Because document[0] is the word_list as we declare it on line number 40
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] #This will lemmatize the word which are in word_patterns in document[0] and convert it in lower case
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = list(output_empty)#copying output_empty to another variable
    output_row[classes.index(document[1])] = 1

    #Last Append everythin in training
    training.append([bag,output_row])
#10.Shuffling the training data
random.shuffle(training)
#11.Converting itself into numpy array for Feeding to neural network
training = np.array(training)

# 12. dividing data into x and y training
train_x = list(training[:,0])
train_y = list(training[:,1])




