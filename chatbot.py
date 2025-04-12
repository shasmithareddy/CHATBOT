import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON
intents = json.loads(open('C:/Users/shasm/Downloads/chatbot/intents.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load trained model
model = load_model('chatbot_model.h5')

# Tokenize and lemmatize sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict intent using the model
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Get appropriate response from intents JSON
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that. Try asking something else."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Hmm... I'm still learning. Can you rephrase?"

# Start chatbot
print("🤖 ChatBot is running! Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        message = input("You: ")
        if message.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! 👋 Take care.")
            break
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(f"Bot: {res}")
    except Exception as e:
        print(f"Bot: Oops! Something went wrong: {e}")
