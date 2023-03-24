from torch import nn

import cv2
import os

from models.lift_splat_shoot import LiftSplatShoot
from models.lift_splat_shoot_ensemble import LiftSplatShootEnsemble
from models.lift_splat_shoot_gpn import LiftSplatShootGPN
from models.lift_splat_shoot_dropout import LiftSplatShootDropout

from tools.loss import *
from tools.uncertainty import *


colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])


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


def get_step(preds, labels, activation, loss_fn, type):
    if type == 'postnet_uce' or type == 'postnet_uce_cnn' or type == 'enn_uce':
        preds = activation(preds)
        loss = loss_fn(preds, labels)
    elif type == 'baseline_ce' or type == 'dropout_ce' or type == 'ensemble_ce':
        loss = loss_fn(preds, labels)
        preds = activation(preds, dim=1)
    elif type == 'enn_ce' or type == 'postnet_ce' or type == 'postnet_ce_cnn':
        preds = activation(preds)
        loss = loss_fn(preds.log(), torch.argmax(labels, dim=1))

    return preds, loss


def get_model(type, num_classes, device):
    if type == 'baseline_ce':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShoot(outC=num_classes)
    elif type == 'enn_ce':
        activation = activate_uncertainty
        loss_fn = torch.nn.NLLLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShoot(outC=num_classes)
    elif type == 'enn_uce':
        activation = activate_gpn
        loss_fn = uce_loss
        model = LiftSplatShoot(outC=num_classes)
    elif type == 'postnet_ce':
        activation = activate_gpn
        loss_fn = torch.nn.NLLLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif type == 'postnet_ce_cnn':
        activation = activate_gpn
        loss_fn = torch.nn.NLLLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShootGPN(outC=num_classes)
    elif type == 'postnet_uce':
        activation = activate_gpn
        loss_fn = uce_loss
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif type == 'postnet_uce_cnn':
        activation = activate_gpn
        loss_fn = uce_loss
        model = LiftSplatShootGPN(outC=num_classes)
    elif type == 'dropout_ce':
        activation = torch.softmax
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShootDropout(outC=num_classes)
    elif type == 'ensemble_ce':
        activation = torch.softmax
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShootEnsemble(outC=num_classes)
    else:
        raise ValueError("Please pick a valid model type.")

    return activation, loss_fn, model


def map(img):
    dense = img.detach().cpu().numpy().argmax(-1)
    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color
    return rgb


def save_pred(pred, labels, out_path):
    pred = map(pred[0].permute(1, 2, 0))
    labels = map(labels[0].permute(1, 2, 0))

    cv2.imwrite(os.path.join(out_path, "pred.jpg"), pred)
    cv2.imwrite(os.path.join(out_path, "label.jpg"), labels)
