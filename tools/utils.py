import torch.nn

from datasets.nuscenes import *
from models.cvt.cross_view_transformer import *
from models.lss.lift_splat_shoot import LiftSplatShoot
from models.lss.lift_splat_shoot_ensemble import LiftSplatShootEnsemble
from models.lss.lift_splat_shoot_gpn import LiftSplatShootGPN
from models.lss.lift_splat_shoot_dropout import LiftSplatShootDropout
import matplotlib.pyplot as plt
import matplotlib as mpl

from tools.loss import *
from tools.uncertainty import *


colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])


H = 128
W = 352

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def draw(preds, imgs, rots, trans, intrins, post_rots, post_trans, labels):
    val = 0.01
    fH, fW = (128, 352)
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    pmax = torch.argmax(preds, dim=1).cpu()

    for si in range(imgs.shape[0]):
        pts = [[], [], []]
        l = []

        for i in range(200):
            for j in range(200):
                if pmax[si, i, j] == 3:
                    continue
                pts[0].append(i / 2 - 50)
                pts[1].append(j / 2 - 50)
                pts[2].append(0)
                l.append(pmax[si, i, j])

        l = np.array(l)
        pts = torch.tensor(pts)

        plt.clf()
        for imgi, img in enumerate(imgs[si]):
            ego_pts = ego_to_cam(pts, rots[si, imgi], trans[si, imgi], intrins[si, imgi])
            plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)
            mask = get_only_in_img_mask(plot_pts, H, W)

            rgb = np.zeros((*l.shape, 3))
            for label, color in enumerate(colors):
                rgb[l == label] = color

            ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
            showimg = denormalize_img(img)

            plt.imshow(showimg)
            plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=rgb[mask, :]/255.0,
                        s=5, alpha=0.2)

            plt.axis('off')
            # plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

        ax = plt.subplot(gs[0, :])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        plt.setp(ax.spines.values(), color='b', linewidth=2)
        plt.imshow(map(pmax, max=True)[0].astype(np.uint8))

        plt.savefig("mapped.jpg")
        plt.close()


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


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


def get_step(preds, labels, activation, loss_fn, type):
    if type == 'postnet' or type == 'enn':
        preds = activation(preds)
        loss = loss_fn(preds, labels)
    elif type == 'baseline' or type == 'dropout' or type == 'ensemble':
        loss = loss_fn(preds, labels)
        preds = activation(preds, dim=1)

    return preds, loss


backbones = {
    'lss': [LiftSplatShoot, LiftSplatShootGPN, LiftSplatShootDropout, LiftSplatShootEnsemble],
    'cvt': [CrossViewTransformer, CrossViewTransformerGPN, CrossViewTransformerDropout, CrossViewTransformerEnsemble]
}


def get_model(type, backbone, num_classes, device):
    if type == 'baseline':
        activation = torch.softmax
        # loss_fn = torch.nn.CrossEntropyLoss().cuda(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 3.0, 1.0])).cuda(device)
        # loss_fn = SigmoidFocalLoss().cuda(device)
        # loss_fn = torch.nn.BCEWithLogitsLoss().cuda(device)
        model = backbones[backbone][0](outC=num_classes)
    elif type == 'enn':
        activation = activate_uncertainty
        loss_fn = uce_loss
        model = backbones[backbone][0](outC=num_classes)
    elif type == 'postnet':
        activation = activate_gpn
        loss_fn = uce_loss
        model = backbones[backbone][1](outC=num_classes)
    elif type == 'dropout':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 3.0, 1.0])).cuda(device)
        model = backbones[backbone][2](outC=num_classes)
    elif type == 'ensemble':
        activation = torch.softmax
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 3.0, 1.0])).cuda(device)
        model = backbones[backbone][3](outC=num_classes)
    else:
        raise ValueError("Please pick a valid model type.")

    return activation, loss_fn, model


def map(img, max=False):
    if not max:
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

