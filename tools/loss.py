import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.distributions as dist
import cv2
from tools.uncertainty import activate_uce, vacuity
import torchvision

from einops import rearrange


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        y = y.argmax(dim=1)
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class UCELossReg(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
        l=0.000005
    ):
        super().__init__()

        self.weights = weights
        self.loss_fn = UCELoss(weights)
        self.l = l

    def forward(self, alpha, y, ood):
        mask = ood.unsqueeze(0).repeat(4, 1, 1, 1).bool()

        alpha = alpha.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        alpha_masked = alpha[mask].view(4, -1)

        pred = dist.Dirichlet(alpha_masked)
        target = dist.Dirichlet(torch.ones_like(alpha_masked))

        reg = dist.kl.kl_divergence(target, pred)

        alpha = alpha[~mask].view(1, 4, -1)
        y = y[~mask].view(1, 4, -1)

        A = self.loss_fn(alpha, y)

        return A + torch.mean(reg) * self.l


hw = [
    (56, 120),
    (14, 30),
]


def unravel_index(indices, shape):
    unraveled_indices = []
    for dim in reversed(shape):
        unraveled_indices.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(unraveled_indices))


def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones_like(alpha, dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


class UCELossRegMap(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
        l=0.0001
    ):
        super().__init__()

        self.weights = weights
        self.loss_fn = UCELoss(weights)
        self.l = l
        self.bce = torch.nn.BCELoss()

    def forward(self, alpha, y, ood, ood_cam, atts):
        A = self.loss_fn(alpha, y)

        mask = torch.zeros((alpha.shape[0], 25, 25)).to(ood.device)

        rearranged_atts = []
        for i, att in enumerate(atts):
            r = rearrange(att, '(b m) (H W) (n h w) -> b m H W n h w', n=6, m=4, H=25, W=25, h=hw[i][0],
                                      w=hw[i][1])
            rearranged_atts.append(r)

        att = rearranged_atts[0]
        mean_att = torch.mean(att, dim=1)

        for i in range(alpha.shape[0]):
            for r in range(25):
                for c in range(25):
                    att_map = mean_att[i, r, c]
                    l = att_map.argmax()
                    mc, my, mx = unravel_index(l, att_map.shape)

                    mask[i, r, c] = ood_cam[i, mc, my, mx]

        mask.requires_grad = True
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(200, 200))
        cv2.imwrite("mask.jpg", mask[0,0].detach().cpu().numpy()*255)

        reg = self.bce(mask.float(), y[:,0,:,:].float().unsqueeze(1))

        return reg


class UCELoss(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
    ):
        super().__init__()

        self.weights = weights

    def forward(self, alpha, y, epoch_num):
        S = torch.sum(alpha, dim=1, keepdim=True)

        B = y * (torch.digamma(S + 1e-10) - torch.digamma(alpha + 1e-10) + 1e-10)
        # (bsx x 4 x 200 x 200)

        if self.weights is not None:
            for i in range(self.weights.shape[0]):
                B[:, i] *= self.weights[i]

        A = torch.sum(B, dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / 10, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, 4, device=alpha.device)

        return (A + kl_div).mean()


class FocalUCELoss(torch.nn.Module):
    def __init__(
        self,
        n=2,
        weights: Optional[Tensor] = None,
    ):
        super().__init__()

        self.weights = weights
        self.n = n

    def forward(self, alpha, y):
        S = torch.sum(alpha, dim=1, keepdim=True)

        a0 = S
        aj = torch.gather(alpha, 1, torch.argmax(y, dim=1, keepdim=True))

        B = (gamma(a0 - aj + self.n) * gamma(a0)
             / (gamma(a0 + self.n) * gamma(a0 - aj))) \
            * torch.digamma(a0 + self.n + 1e-10) - torch.digamma(aj + 1e-10)

        if self.weights is not None:
            for i in range(self.weights.shape[0]):
                B[:, i, :, :] *= self.weights[i]

        A = torch.sum(B, dim=1, keepdim=True)

        return A.mean()


def gamma(x):
    return torch.exp(torch.lgamma(x))


def scatter(x, classes, colors):
    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                          data=np.column_stack((x, colors)))
    cps_df['target'] = cps_df['target'].astype(int)
    cps_df.head()
    grid = sns.FacetGrid(cps_df, hue="target", height=10, legend_out=False)
    plot = grid.map(plt.scatter, 'CP1', 'CP2')
    plot.add_legend()
    for t, l in zip(plot._legend.texts, classes):
        t.set_text(l)

    return plot