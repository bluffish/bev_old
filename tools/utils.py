import torch
import cv2
import os

import numpy as np


def get_iou(preds, labels):
    classes = preds.shape[1]
    intersect = [0]*classes
    union = [0]*classes

    with torch.no_grad():
        for i in range(classes):
            pred = (preds[:, i, :, :] > 0.5)
            tgt = labels[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return intersect, union

colors = [
    [255, 0, 0],
    [0, 0, 255],
    [0, 0, 0]
]

def save_pred(pred, labels, out_path):
    cv2.imwrite(os.path.join(out_path, "pred.jpg"), np.array(pred.detach().cpu()[0].permute(1, 2, 0)) * 255)
    cv2.imwrite(os.path.join(out_path, "label.jpg"), np.array(labels.detach().cpu()[0].permute(1, 2, 0)) * 255)
