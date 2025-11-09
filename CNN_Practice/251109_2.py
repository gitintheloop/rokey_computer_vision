# 251109_2.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Dataset Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size = 64
# MNIST dataset download 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# train_dataset[0]
# train_dataset[0][0].shape

# Model Definition 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # self.conv_pool_fc_relu_flatten_stack = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.Linear(64 * 7 * 7, 128),
        #     nn.ReLU(),
        #     nn.Flatten()   
        # )   

    def forward(self, x):
        # TO-DO: 첫 번째 convolution layer -> ReLU -> Max pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # TO-DO: 두 번째 convolution layer -> ReLU -> Max pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # TO-DO: view 함수로 특징맵을 1차원으로 펼치기(위 init에서 self.flatten도 사용 가능)
        x = self.flatten(x)
        # TO-DO: 첫 번째 fc-layer -> ReLU
        x = self.fc1(x)
        x = self.relu(x)
        # TO-DO: 두 번째 fc-layer -> 최종 logit 출력
        x = self.fc2(x)
        return x

# device 저장. gpu 사용 가능하면 gpu로 설정.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 인스턴스 생성
model = SimpleCNN().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss() # 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam 옵티마이저 사용, 학습률 0.001

# Epoch 수 결정
epochs = 10

import torchsummary
torchsummary.summary(model, (1,28,28), device=str(device))

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nAccuracy: {100 * correct / total:.2f}%")

train_model(model, train_loader, criterion, optimizer, epochs)

evaluate_model(model, test_loader)