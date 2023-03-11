import torch


def loss_reduce(
        loss: torch.Tensor,
        reduction: str = 'sum') -> torch.Tensor:
    """utility function to reduce raw losses
    Args:
        loss (torch.Tensor): raw loss which should be reduced
        reduction (str, optional): reduction method ('sum' | 'mean' | 'none')
    Returns:
        torch.Tensor: reduced loss
    """

    if reduction == 'sum':
        return loss.sum()

    if reduction == 'mean':
        return loss.mean()

    if reduction == 'none':
        return loss

    raise ValueError(f'{reduction} is not a valid value for reduction')


def activate_gpn(alpha):
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    return prob


def uce_loss(
        alpha: torch.Tensor,
        y: torch.Tensor,
        reduction: str = 'mean') -> torch.Tensor:
    """utility function computing uncertainty cross entropy /
    bayesian risk of cross entropy
    Args:
        alpha (torch.Tensor): parameters of Dirichlet distribution
        y (torch.Tensor): ground-truth class labels (not one-hot encoded)
        reduction (str, optional): reduction method ('sum' | 'mean' | 'none').
            Defaults to 'sum'.
    Returns:
        torch.Tensor: loss
    """
    S = torch.sum(alpha, dim=1, keepdim=True)

    B = y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10)

    B[:, 0, :, :] *= 2
    B[:, 2, :, :] *= 4

    A = torch.sum(B, dim=1, keepdim=True)

    return loss_reduce(A, reduction=reduction)