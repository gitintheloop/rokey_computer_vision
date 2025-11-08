#Batch_Size.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms # Preprocessing
# torchvision.transforms transforms a raw image into tensor and normalizes it.
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('='*50)
# 1. Transform & Dataset
print('='*50)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # mean 0.5 std 0.5
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Small Batch vs Large Batch
trainloader_small = DataLoader(trainset, batch_size=8, shuffle=True)
trainloader_large = DataLoader(trainset, batch_size=256, shuffle=True)

print('='*50)
# 2. Simple CNN model definition
print('='*50)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extraction
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Linear(32*8*8,10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
print('='*50)
# 3. Training Fuction Definition
print('='*50)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

print('='*50)
# 4. Comparison between small batch and large batch
print('='*50)

cirterion = nn.CrossEntropyLoss()

# Small Batch
model_small = SimpleCNN()
optimizer_small = optim.Adam(model_small.parameters(), lr=0.001)
loss_small = train_one_epoch(model_small, trainloader_small, optimizer_small, cirterion)

# Large Batch
model_large = SimpleCNN()
optimizer_large = optim.Adam(model_large.parameters(), lr=0.001)
loss_large = train_one_epoch(model_large, trainloader_large, optimizer_large, cirterion)

print(f"Small batch (8) Loss: {loss_small:.4f}")
print(f"Large batch (8) Loss: {loss_large:.4f}")