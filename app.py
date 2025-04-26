from flask import Flask, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

with open("intents.json", "r") as file:
    intents = json.load(file)

fallback_responses = [
    "Could you please describe your issue more clearly?",
    "I'm not sure I understood. Can you specify more?",
    "Sorry, I didn't get that. Are you asking about legal rights or something else?"
]

def preprocess_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return words

def match_intent(user_input):
    user_words = preprocess_sentence(user_input)
    best_match = None
    highest_overlap = 0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = preprocess_sentence(pattern)
            common_words = set(user_words) & set(pattern_words)
            
            if len(common_words) > highest_overlap:
                highest_overlap = len(common_words)
                best_match = intent['tag']

    return best_match

def generate_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return random.choice(fallback_responses)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    intent_tag = match_intent(user_input)
    if intent_tag:
        bot_response = generate_response(intent_tag)
    else:
        bot_response = random.choice(fallback_responses)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
