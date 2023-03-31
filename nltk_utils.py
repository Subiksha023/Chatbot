#utils means commonly used fns that r used repeatedly

import nltk
nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in tokenized_sentence]
  """
  sentence = ["Hi","How", "are", "you"]
  words = ["Hi", "good", "cool", "bye", "superb"]
  bag = [1 , 0 , 0 , 0, 0]
  """
  # we create a array of zeros with the size of the 'words' list and then if the sentence element is in the words, then we change the value in bag to 1
  bag = np.zeros(len(all_words), dtype = np.float32)

  for idx, w in enumerate (all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0
  return bag


"""
a = "How is my dress!"
print(a)
print(tokenize(a))

words = ["Organizes", "organizes" , "organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
"""