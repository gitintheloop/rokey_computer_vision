import torch.nn as nn
import torch 

# Input Channels (grayscale image)
in_channels = 1
# Output Channels (number of filters)
kernel_size_conv = 5
kernel_size_pool = 2
stride_pool = 2

conv1 = nn.Conv2d(
    in_channels = 1,
    out_channels = 10,
    kernel_size = 5
)

relu = nn.ReLU()

pool1 = nn.MaxPool2d(
    kernel_size = 2,
    stride = 2
)

print(f"Conv2d Layer: {conv1}")
print(f"MaxPool2d Layer: {pool1}")

dummy_image = torch.randn(100, 1, 28, 28)

c_out = conv1(dummy_image)
r_out = relu(c_out)
p_out = pool1(r_out)

print(f"\nOriginal shape: {dummy_image.shape}")
print(f"After Conv1: {c_out.shape}")
print(f"After MaxPool1: {p_out.shape}")