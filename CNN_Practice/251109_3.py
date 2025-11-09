# 251109_3.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchviz 
import torchinfo 
import torchvision.transforms as transforms

# Drop out
torch.manual_seed(123)
inputs = torch.randn(1, 10)
print("Original Inputs:")
print(inputs)

# Definition Drop Out Function
dropout = nn.Dropout(0.5)

# Training Phase Motion
dropout.train() # training mode
print(f"Is in training mode? : {dropout.training}")
outputs = dropout(inputs)
print("Outputs in Training mode: ")
print(outputs)

# Prediction Phase Motion
dropout.eval() # evaluation mode # DropOut Inactivation
print(f"Is in training mode : {dropout.training}")
outputs = dropout(inputs)
print("Outputs in Eval Mode:")
print(outputs) 

print("="*100)
# Data Augmentation
# Activating Data Augmentation using Transforms
# Training Data : Normalization, Transpos, RandomErasing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio = (0.3, 3.3), value=0, inplace=True)])
print(transform_train)