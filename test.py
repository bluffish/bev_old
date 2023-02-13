import torch

x = torch.randn((1, 3, 2, 2))
print(x)
print(x.view(3, 4))