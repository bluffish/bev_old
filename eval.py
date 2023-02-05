import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

from datasets.nuscenes import compile_data
from models.lift_splat_shoot import LiftSplatShoot
from models.lift_splat_shoot_gpn import LiftSplatShootGPN

from datasets.carla import CarlaDataset
from tools.utils import *
from tools.uncertainty import *
from tools.gpn_loss import *

import argparse
import yaml

torch.multiprocessing.set_sharing_strategy('file_system')


def eval(
        config
):

    if config['dataset'] == 'carla':
        train_dataset = CarlaDataset(os.path.join(config['data_path'], "train/"))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'],
                                                   drop_last=True)

        val_dataset = CarlaDataset(os.path.join(config['data_path'], "val/"))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config['batch_size'],
                                                 shuffle=True,
                                                 num_workers=config['num_workers'],
                                                 drop_last=True)
    elif config['dataset'] == 'nuscenes':
        train_loader, val_loader = compile_data("mini", "../data/nuscenes/", config['batch_size'], config['num_workers'])

    gpus = config['gpus']
    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')
    num_classes = 4

    if config['type'] == 'baseline_ce':
        activation = torch.softmax
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'baseline_uce':
        activation = activate_uncertainty
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'postnet_ce':
        activation = activate_gpn
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif config['type'] == 'postnet_uce':
        activation = activate_gpn
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif config['type'] == 'postnet_uce_cnn':
        activation = activate_gpn
        model = LiftSplatShootGPN(outC=num_classes)
    else:
        raise ValueError("Please pick a valid model type.")

    model = nn.DataParallel(model, device_ids=gpus).to(device).eval()
    model.load_state_dict(torch.load(config['model_path']))

    print("--------------------------------------------------")
    print(f"Starting eval on {config['type']} model")
    print(f"Using GPUS: {gpus}")
    print("Training using CARLA ")
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    print('running eval...')

    total_intersect = [0]*num_classes
    total_union = [0]*num_classes

    y_true = []
    y_scores = []

    out_path = "./outputs/"+config['type']

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):

            preds = model(imgs,
                          rots,
                          trans,
                          intrins,
                          post_rots,
                          post_trans)

            labels = labels.to(device)

            uncert = entropy(preds)

            if config['type'] == 'postnet_uce' \
                    or config['type'] == 'postnet_uce_cnn' \
                    or config['type'] == 'baseline_uce':
                preds = activation(preds)
            elif config['type'] == 'baseline_ce':
                preds = activation(preds, dim=1)
            elif config['type'] == 'postnet_ce':
                preds = activation(preds)

            intersect, union = get_iou(preds, labels)

            for i in range(num_classes):
                total_intersect[i] += intersect[i]
                total_union[i] += union[i]

            save_pred(preds, labels, out_path)
            plt.imsave(os.path.join(out_path, "uncertainty_map.jpg"), plt.cm.jet(uncert[0][0]))

            preds = preds[:, 0, :, :].ravel()
            labels = labels[:, 0, :, :].ravel()
            uncert = torch.tensor(uncert).ravel()

            vehicle = np.logical_or(preds.cpu() > 0.5, labels.cpu() == 1).bool()

            preds = preds[vehicle]
            labels = labels[vehicle]
            uncert = uncert[vehicle]

            pred = (preds > 0.5)
            tgt = labels.bool()
            intersect = (pred == tgt).type(torch.int64)

            y_true += intersect.tolist()
            # y_true += labels.cpu().tolist()
            uncert = -uncert
            y_scores += uncert.tolist()

    iou = [0]*num_classes

    for i in range(num_classes):
        iou[i] = total_intersect[i]/total_union[i]

    print('iou: ' + str(iou))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.ylim([0, 1.05])
    plt.ylim([0, 1.05])

    roc_display.plot(ax=ax1, label=config['type'])
    pr_display.plot(ax=ax2, label=config['type'])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    plt.savefig(os.path.join(out_path, "combined.jpg"))
    print(f"AUPR: {pr} AUROC: {auc_score}")

    return roc_display, pr_display, auc_score, pr, iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    roc_display, pr_display, auc_score, pr, iou = eval(config)
