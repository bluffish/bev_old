import torch


alpha = torch.randn((1, 4, 3, 3))

print(alpha)

print(alpha.permute(0, 2, 3, 1).view(-1, 4))
