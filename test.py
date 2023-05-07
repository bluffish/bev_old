import torch

def unravel_index(indices, shape):
    unraveled_indices = []
    for dim in reversed(shape):
        unraveled_indices.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(unraveled_indices))

print(unravel_index(torch.tensor(5), torch.tensor((1, 2, 4))))