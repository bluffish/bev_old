import torch.nn

from datasets.nuscenes import *
from models.cvt.cross_view_transformer import *
from models.lss.lift_splat_shoot import LiftSplatShoot, LiftSplatShootENN
from models.lss.lift_splat_shoot_ensemble import LiftSplatShootEnsemble
from models.lss.lift_splat_shoot_gpn import LiftSplatShootGPN
from models.lss.lift_splat_shoot_dropout import LiftSplatShootDropout

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

    pmax = torch.argmax(preds, dim=1)

    with torch.no_grad():
        for i in range(classes):
            pred = (pmax == i)

            tgt = labels[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return intersect, union


backbones = {
    'lss': [LiftSplatShoot, LiftSplatShootENN, LiftSplatShootGPN, LiftSplatShootDropout, LiftSplatShootEnsemble],
    'cvt': [CrossViewTransformer, CrossViewTransformerENN, CrossViewTransformerGPN, CrossViewTransformerDropout,
            CrossViewTransformerEnsemble]
}


def get_model(type, backbone, num_classes, device):
    weights = torch.tensor([3.0, 1.0, 2.0, 1.0]).to(device)

    if type == 'baseline':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights).cuda(device)
        model = backbones[backbone][0](outC=num_classes)
    elif type == 'enn':
        activation = activate_uce
        loss_fn = UCELoss(weights=weights).cuda(device)
        model = backbones[backbone][1](outC=num_classes)
    elif type == 'postnet':
        activation = activate_uce
        loss_fn = UCELoss(weights=weights).cuda(device)
        model = backbones[backbone][2](outC=num_classes)
    elif type == 'dropout':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights).cuda(device)
        model = backbones[backbone][3](outC=num_classes)
    elif type == 'ensemble':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights).cuda(device)
        model = backbones[backbone][4](outC=num_classes)
    else:
        raise ValueError("Please pick a valid model type.")

    print(f"Loss {loss_fn}")
    print(f"Activation {activation}")

    return activation, loss_fn, model


def map(img, m=False):
    if not m:
        dense = img.detach().cpu().numpy().argmax(-1)
    else:
        dense = img.detach().cpu().numpy()

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color
    return rgb


def save_pred(pred, labels, out_path):
    pred = map(pred[0].permute(1, 2, 0))
    labels = map(labels[0].permute(1, 2, 0))

    cv2.imwrite(os.path.join(out_path, "pred.jpg"), pred)
    cv2.imwrite(os.path.join(out_path, "label.jpg"), labels)

    return pred, labels
