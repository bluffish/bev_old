import numpy as np
import torch
from tools.uncertainty import activate_uce
from tools.utils import save_pred
import einops

alpha = torch.load("alpha.pt")
alpha = alpha.view(4, -1)
alpha = alpha[0]
alpha = alpha.view(8, 4, 200, 200)
print(alpha.shape)

pred = activate_uce(alpha)

p, l = save_pred(pred, pred, './')
