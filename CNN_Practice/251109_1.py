# 251109_1.py

import torch

cnn1 = torch.nn.Conv2d(2, 3, kernel_size=4, stride=1, padding=1)

# Weight, Bias
cnn1.weight.data = torch.randint(low=1, high=5, size=(3, 2, 2, 2),dtype=torch.int).float()
cnn1.bias.data = torch.tensor([1,2,3]).float()

# Random input generation
input = torch.randint(low=0, high=3, size=(2,5,5), dtype=torch.int).float()
print("input tensor: ", input)

print(input.shape)
print("=" * 100)

print(cnn1.weight)
print(cnn1.bias)
print("=" * 100)

print(cnn1.weight.shape)
print(cnn1.bias.shape)
print("=" * 100)

output = cnn1(input)
print(output.shape)

input = torch.tensor([[[12 , 20 , 30, 0],
                       [8  , 12 , 2 , 0],
                       [34 , 70 , 37, 4],
                       [112, 100, 25, 12]]])

MaxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
Output = MaxPool(input)
print(Output)