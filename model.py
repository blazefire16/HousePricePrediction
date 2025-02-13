import torch
import torch.nn as nn

class HousePriceModel(nn.Module):
    def __init__(self, input_size):  # input_size is the number of features
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input size should be `input_size`, which is 5
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # This will now work with 5 input features
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

