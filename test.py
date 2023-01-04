import torch

x = torch.randn(1, 3, 2, 2)

print(x)

flat = x.reshape(3, -1)

print(flat)
