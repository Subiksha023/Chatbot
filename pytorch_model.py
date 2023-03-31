#PyTorch model and training

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes ):
    super(NeuralNet,self).__init__()
  # torch.nn.Linear(in_features(int), out_features(int), bias=True, device=None, dtype=None)
    self.l1 = nn.Linear(input_size, hidden_size )
    self.l2 = nn.Linear(hidden_size, hidden_size )
    self.l2 = nn.Linear(hidden_size, num_classes )
    self.relu = nn.ReLU()

# implementing forward pass
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    #no activation or softmax, we apply corss entropy loss
    return out

