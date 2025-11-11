# 251111.py

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)
base_lr = 0.01

fc_params = set(model.fc.parameters())
layer4_params = set(model.layer4.parameters())

other_params = [param for param in model.parameters() if
                (param not in fc_params) and (param not in layer4_params)]

params_to_optimize = [
    {'params': model.fc.parameters(), 'lr':base_lr},
    {'params': model.layer4_parameters(), 'lr':base_lr*0.1 },
    {'params': other_params, 'lr': base_lr*0.01}
]

optimizer = optim.SGD(params_to_optimize, momentum=0.9)

print("옵티마이저가 학습할 파라미터 그룹 : ")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"\n[그룹 {i+1}]")
    print(f"Learning Rate: {param_group['lr']:.4f}")
    print(f"Numbers of Parameters: {len(param_group['params'])}")

    