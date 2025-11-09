# CNN

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data", # dataset 경로지정
    train = True, # True로 선언시 학습 데이터를 가져옴
    download = True, # root에 데이터셋이 없을 경우 download를 함.
    transform = ToTensor() # tensor 형태로 변환시킴
)

test_data =  datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y : ", y.shape, y.dtype)
    print("label: ", y)
    break

# 4차원 입력을 받아 가중치 연산과bias 더하기를 통해 3차원 출력으로 변환
fc1 = nn.Linear(4, 3)

# nn.Linear의 파라미터는 weight와 bias 2가지 존재
fc1.weight.data = torch.tensor([[0.1, 0.3, 0.1, 0.1],
                                [0.1, 0.3, 0.4, 0.1],
                                [0.3, 0.3, 0.1, 0.2]], dtype=torch.float)
fc1.bias.data = torch.tensor([3, 6, 9], dtype=torch.float)

input = torch.tensor([1, 2, 3, 4], dtype=torch.int).float()
print("입력: ", input)
print("=" * 100)
print("fc1 레이어의 weight: ", fc1.weight.data)
print("fc1 레이어의 weight shape: ", fc1.weight.shape)

print("=" * 100)
print("fc1 레이어의 bias: ", fc1.bias.data)
print("fc1 레이어의 bias shape: ", fc1.bias.shape)

print("=" * 100)
output = fc1(input)
print("fc1의 출력: ", output)
print("fc1의 출력 shape: ", output.shape)

print("=" * 100)
# 학습에 사용할 CPU, GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Model Definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # Input 28*28, Output 512 
            nn.ReLU(),              # Activation Function
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

        # self.linear1 = nn.Linear(28*28, 512)
        # self.linear2 = nn.Linear(512, 512)
        # self.linear3 = nn.Linear(512, 10)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) # 28*28 -> 1*784
        logtis = self.linear_relu_stack(x)
        return logtis
    
    # def forward(self, x):
    #     x = self.flatten(x)
    #     x = self.linear1(x)
    #     x = self.relu(x)
    #     x = self.linear2(x)
    #     x = self.relu(x)
    #     x = self.linear3(x)
    #     return x

model = NeuralNetwork().to(device)
print(model)