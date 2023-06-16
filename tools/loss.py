import torch
from torch import Tensor
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.distributions as dist


class UCELoss(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
        num_classes=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights.view(1, self.num_classes, 1, 1)

    def loss(self, alpha, y):
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10) * self.weights, dim=1, keepdim=True)

        return A.mean()

    def forward(self, alpha, y):

        return self.loss(alpha, y)


class UCELossReg(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
        num_classes=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = UCELoss(weights=weights, num_classes=num_classes)

        self.l = .00001

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

        # return A
        return A + torch.mean(reg) * self.l


class CELoss(torch.nn.Module):
    def __init__(
            self,
            weights: Optional[Tensor] = None,
            num_classes=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights.view(1, self.num_classes, 1, 1)

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, pred, target):
        if pred.ndim > 4:
            pred = torch.mean(pred, dim=0)

        return self.loss_fn(pred, target)


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