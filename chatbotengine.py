# chatbot_engine.py

import random
import json
import numpy as np
import torch
import torch.nn as nn
from preprocessing import bag_of_words, tokenize

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.relu(self.l1(x))
        out = torch.relu(self.l2(out))
        out = self.l3(out)
        return out

class Chatbot:
    def __init__(self, intents_file, model_data):
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)

        data = torch.load(model_data)

        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model_state = data["model_state"]

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def get_response(self, sentence):
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])

        return "I'm sorry, I didn't understand. Could you please rephrase?"

