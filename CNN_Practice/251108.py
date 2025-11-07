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

print("=======================================================")

class PartialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10*12*12, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50,10)

    def forward(self, x_pool2):
        x_flat = x_pool2.view(x_pool2.size(0),-1)

        x = self.fc1(x_flat)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x_flat
    
net_cnn = PartialCNN()
dummy_pool_out = torch.randn(100, 10, 12, 12)
final_output, flat_output = net_cnn(dummy_pool_out)

print(f"Model : \n{net_cnn}")
print(f"\nShape before Flatten: {dummy_pool_out.shape}")
print(f"\nShape after Flatten: {flat_output.shape}")
print(f"Final Output Shape: {final_output.shape}")