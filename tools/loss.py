import torch
from torch import Tensor
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


class UCELoss(torch.nn.Module):
    def __init__(
        self,
        weights: Optional[Tensor] = None,
        num_classes=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights.view(1, self.num_classes, 1, 1)

    def forward(self, alpha, y, epoch_num):
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10) * self.weights, dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / 8, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device=alpha.device)

        return (A + kl_div).mean()


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