import torch


def activate_gpn(alpha):
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    return prob


def loss_reduce(
        loss: torch.Tensor,
        reduction: str = 'sum') -> torch.Tensor:

    if reduction == 'sum':
        return loss.sum()

    if reduction == 'mean':
        return loss.mean()

    if reduction == 'none':
        return loss

    raise ValueError(f'{reduction} is not a valid value for reduction')


def uce_loss(
        alpha: torch.Tensor,
        y: torch.Tensor,
        reduction: str = 'mean') -> torch.Tensor:

    S = torch.sum(alpha, dim=1, keepdim=True)

    B = y * (torch.digamma(S + 1e-10) - torch.digamma(alpha + 1e-10) + 1e-10)

    B[:, 0, :, :] *= 2
    B[:, 2, :, :] *= 4

    A = torch.sum(B, dim=1, keepdim=True)

    return loss_reduce(A, reduction=reduction)