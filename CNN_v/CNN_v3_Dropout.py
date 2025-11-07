# CNN_v3_Dropout.py
import torch
import torch.nn as nn


class CNN_v3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)        
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)

        # Definition of three Dropouts layers with different ratios
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        self.features = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu, self.maxpool,
            self.dropout1, # Append after the 1st MaxPool layer
            self.conv3, self.relu,
            self.conv4, self.relu, self.maxpool,
            self.dropout2, # Append after the 2nd MaxPool layer
            self.conv5, self.relu,
            self.conv6, self.relu, self.maxpool,
            self.dropout3, # Append after the 3rd MaxPool layer
        )

        self.classifier = nn.Sequential(
            self.l1, self.relu,
            self.dropout3,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3