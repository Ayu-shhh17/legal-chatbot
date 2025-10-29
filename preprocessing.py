# src/preprocessing.py

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import string

# Download NLTK resources if not already available
nltk.download('punkt', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split a sentence into words/tokens.
    Example: "How are you?" -> ["How", "are", "you", "?"]
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Reduce word to its stem form.
    Example: "Organizing" -> "organ"
    """
    return stemmer.stem(word.lower())

def clean_sentence(sentence):
    """
    Convert to lowercase, remove punctuation.
    Example: "Hello, there!" -> "hello there"
    """
    sentence = sentence.lower()
    return sentence.translate(str.maketrans('', '', string.punctuation))

def bag_of_words(tokenized_sentence, all_words):
    """
    Return a bag of words array:
    Example:
      tokenized_sentence = ["hello", "how", "are", "you"]
      all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
      result = [0, 1, 0, 1, 0, 0, 0]
    """
    # Stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Initialize bag with 0 for each known word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
