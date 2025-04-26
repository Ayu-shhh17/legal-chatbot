# main.py

import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

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
    return "I'm sorry, I couldn't understand your query. Could you please clarify?"

def chatbot_response(user_input):
    if not user_input.strip():
        return "Please type something so I can assist you!"
    
    intent_tag = match_intent(user_input)
    if intent_tag:
        return generate_response(intent_tag)
    else:
        return "I couldn't find a matching answer. Could you rephrase your question?"

if __name__ == "__main__":
    print("Legal Bot: Hello! I am your legal assistant. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Legal Bot: Goodbye! Have a great day!")
            break
        
        response = chatbot_response(user_input)
        print(f"Legal Bot: {response}")
