import torch


mask = torch.tensor([[
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
]])

print(mask.shape)
mask = mask.unsqueeze(1).repeat(1,2,1,1).bool()
print(mask.shape)
alpha = torch.ones_like(mask)
print(alpha.shape)

print(alpha[mask].shape)