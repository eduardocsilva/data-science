import random
import json
import numpy as np

import nltk
#nltk.download('punkt') # Must be called on the first execution
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
    Util Functions
"""
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return PorterStemmer().stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] += 1.0

    return bag

"""
    Model Definition
"""
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.l1(input)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

"""
    Load Model and Hyper Parameters
"""
with open("data.json", "r") as f:
    intents = json.load(f)

FILE = "chat_bot.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NN(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

"""
    Start Chat
"""
bot_name = "Sam"
print("Let's chat! Type 'quit' to exit.")
while True:
    sentence = input("You: ")

    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    bag = bag_of_words(sentence, all_words)
    x = bag.reshape(1, bag.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
