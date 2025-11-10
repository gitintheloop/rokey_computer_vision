

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(3, 6, 5) # (input_channel, output_channel, filter size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Classifier
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
                          # (Batchsize, Channel, Width, Height)  
    def forward(self, x): # x의 shape: [N, 3, 32, 32]
         x = self.pool(F.relu(self.conv1(x)))  # [N, 6, 14, 14]
         x = self.pool(F.relu(self.conv2(x)))  # [N, 16, 5, 5] 

