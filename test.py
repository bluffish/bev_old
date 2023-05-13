import torch
import torch.distributions as dist

from tools.loss import *
l = UCELoss()

torch.manual_seed(0)

a = torch.randn((2, 4, 3, 3)).softmax(dim=1)
y = torch.randn((2, 4, 3, 3)).softmax(dim=1)

for i in range(4):
    a[:, i, :, :] = torch.fill(a[:, i, :, :], i+1)

print(l(a, y))

a = a.permute(1,0,2,3)
y = y.permute(1,0,2,3)

mask = torch.ones((2,3,3)).unsqueeze(0).repeat(4, 1, 1, 1).bool()

a2 = a[mask].view(4, -1)
y2 = y[mask].view(4, -1)

print(a2)

print(l(a[mask].view(1, 4, -1), y[mask].view(1, 4, -1)))

print(torch.mean(dist.kl.kl_divergence(dist.Dirichlet(y2, validate_args=True), dist.Dirichlet(a2, validate_args=True)))/2)
