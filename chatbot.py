import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
lemmatizer = WordNetLemmatizer()
spell_checker = SpellChecker()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    corrected_words = [spell_checker.correction(word) for word in sentence_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in corrected_words]
    return lemmatized_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow= bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Carga del archivo JSON con los intents
with open('intents.json', 'r') as file:
    intents = json.load(file)

while True:
    message = input("Tú: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)