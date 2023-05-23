import torch

t = torch.rand((5, 2, 4, 3, 3))
print(t[0])

t = t.ravel()

t = t.view(5, -1)[0]

print(t.view(2, 4, 3, 3))
