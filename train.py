#creating training data
#JavaScript Object Notation (JSON) is a standard text-based format for representing structured data based on JavaScript object syntax.
#In the json file, intents is the keyword

import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_model import NeuralNet


with open ('intents.json' , 'r') as f:
  intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)
  for pattern in intent['patterns']:
    w = tokenize(pattern)
    all_words.extend(w)                      # we use 'extend' instead of 'append' becoz we don't want it to be array inside an array
    xy.append((w, tag))                      # tokenized pattern and tags are in a tuple now

ignore_punctuations = ['.', ',', '?', '!']
all_words = [stem(w) for w in all_words if w not in ignore_punctuations]

all_words = sorted(set(all_words))            # converting all_words list to a set is a nice trick to remove the duplicate items 
tags = sorted(set(tags)) 

# for classification, we cannot use the strings. so bag_of_words is a technique to convert the strings to numbers
X_train = []                                   
Y_train = []                                   # we save the index of the tag in Y_train
for (pattern_sentence, tag) in xy:
  bag = bag_of_words(pattern_sentence, all_words)
  X_train.append(bag)

  label = tags.index(tag)
  Y_train.append(label)               

#we convert the list X_train and Y_train to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# we do batch training. that is, dividing the dataset into multiple batches and then iterating on each batch.
class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(X_train)
    self.x_data = X_train
    self.y_data = Y_train

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.n_samples

#hyperparameters
batch_size = 8
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

device = torch.device('cuds' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
  for (words, labels) in train_loader:
    words = words.to(device)
    labels = labels.to(device)

    #forward pass
    outputs = model(words)
    loss = criterion(outputs, labels)

    #backward pass and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  if (epoch+1) % 100 == 0:
    print(f'epoch {epoch + 1}/ {num_epochs}, loss = {loss.item():.4f}')
  print(f'final loss , loss = {loss.item():.4f}')

data = {
      "model_state" : model.state_dict(),
      "input_size": input_size,
      "hidden_size": hidden_size,
      "output_size": output_size,
      "all_words" : all_words,
      "tags" : tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'file saved to {FILE}')