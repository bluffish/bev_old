import torch
import numpy as np


def activate_uce(alpha):
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


def entropy_dropout(pred):
    mean = []

    for p in pred:
        mean.append(p.softmax(dim=1))

    mean = torch.mean(mean, dim=0)
    class_num = mean.shape[1]
    prob = mean

    e = - prob * (torch.log(prob) / torch.log(class_num))
    total_un = torch.sum(e, dim=1, keepdim=True)

    return total_un


def dissonance(mean):
    evidence = mean + 1
    alpha = mean + 2
    S = torch.sum(alpha, dim=1, keepdim=True)

    belief = evidence / S
    dis_un = torch.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / (term_bj + 1e-7)
            dis_un[k] += dis_ki

    return dis_un


def Bal(b_i, b_j):
    result = 1 - torch.abs(b_i - b_j) / (b_i + b_j + 1e-7)
    return result


def entropy(pred):
    class_num = pred.shape[1]
    prob = pred.softmax(dim=1) + 1e-10
    e = - prob * (torch.log(prob) / np.log(class_num))
    u = torch.sum(e, dim=1, keepdim=True)

    return u


def vacuity(alpha):
    class_num = alpha.shape[1]
    S = torch.sum(alpha, dim=1, keepdim=True)
    v = class_num / S

    return v
