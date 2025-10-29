import torch
import torch.nn as nn
import json
import random
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()

# -----------------------------
# 1️⃣ Utility Functions
# -----------------------------
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

# -----------------------------
# 2️⃣ Neural Network Definition
# -----------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# -----------------------------
# 3️⃣ Load Model and Data
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as f:
    intents = json.load(f)

FILE = "model/model.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# -----------------------------
# 4️⃣ Chatbot Response Logic
# -----------------------------
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return "I'm not sure I understand. Could you rephrase that?"

# -----------------------------
# 5️⃣ Command Line Chat
# -----------------------------
if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'quit' to exit.")
    while True:
        sentence = input("You: ")
        if sentence.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        resp = get_response(sentence)
        print(f"Chatbot: {resp}")



