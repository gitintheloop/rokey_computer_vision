import torch
import numpy as np
import torch.optim as optim

def mse(Yp, Y):
    loss = ((Yp-Y)**2).mean()
    return loss

Yp_test = torch.tensor([1.0, 2.5, 3.8])
Y_test = torch.tensor([1.2, 2.0, 4.0])

test_loss = mse(Yp_test, Y_test)
print(f"{test_loss:.4f}")

print("="*70)

X = torch.tensor([-5., 5., 0., 2., -2.]).float()
Y = torch.tensor([-6.7, 10.3, -3.3, 5.0, -5.3]).float()

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
lr = 0.001

def pred(x): return W * X + B
def mse(Yp, Y): return ((Yp - Y)**2).mean()

Yp = pred(X)
loss = mse(Yp,Y)
loss.backward()

with torch.no_grad():
    W -= lr*W.grad
    B -= lr*B.grad

W.grad.zero_()
B.grad.zero_()

print("="*70)

print(f"W after Update: {W.item():.4f}")
print(f"B after Update: {B.item():.4f}")
print(f"W current gradient: {W.grad}")
print(f"B current gradient: {B.grad}")

X = torch.tensor([-5., 5., 0., 2., -2.]).float()
Y = torch.tensor([-6.7, 10.3, -3.3, 5.0, -5.3]).float()

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
lr = 0.001

def pred(x): return W * X + B
def mse(Yp, Y): return ((Yp - Y)**2).mean()

optimizer = optim.SGD([W,B],lr)
Yp = pred(X)
loss = mse(Yp, Y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
