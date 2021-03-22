import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    #constructor
    def __init__(self,input_size,layer_size,dropout_rate):
        super().__init__() #superclass constructor
        self.fc1 = nn.Linear(input_size,layer_size)
        self.fc2 = nn.Linear(layer_size,layer_size)
        self.fc3 = nn.Linear(layer_size,1)

        self.dropout = nn.Dropout(p=dropout_rate)
        #self.batchnorm = nn.BatchNorm1d(layer_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.batchnorm(x)
        x = F.relu(self.fc2(x))
        #x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
