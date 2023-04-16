from time import time

from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.utils import *
import torch
import torch.nn as nn

import argparse
import yaml
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(0)


def get_val(model, val_loader, device, loss_fn, activation, num_classes):
    total_loss = 0.0
    iou = [0.0] * num_classes

    y_true_m = []
    y_score_m = []
    c = 0

    with torch.no_grad():
        for (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels) in tqdm(val_loader):
            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            labels = labels.to(device)

            if config['type'] == 'baseline' or config['type'] == 'dropout' or config['type'] == 'ensemble':
                uncertainty = entropy(preds)
            if config['type'] == 'enn' or config['type'] == 'postnet':
                uncertainty = dissonance(preds)

            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            total_loss += loss * preds.shape[0]
            intersection, union = get_iou(preds, labels)

            for i in range(0, num_classes):
                if union[0] == 0:
                    iou[i] += 1.0
                else:
                    iou[i] += intersection[i] / union[i] * preds.shape[0]

            if c < 100:
                pmax = torch.argmax(preds, dim=1).cpu()
                lmax = torch.argmax(labels, dim=1).cpu()

                mask = np.logical_or(pmax == 0, lmax == 0).bool()

                p = pmax[mask].ravel()
                l = lmax[mask].ravel()
                u = torch.tensor(uncertainty[:, 0, :, :][mask]).ravel()

                intersect = p != l

                y_true_m += intersect
                y_score_m += u

                c += preds.shape[0]

    iou = [i / len(val_loader.dataset) for i in iou]

    auroc = roc_auc_score(y_true_m, y_score_m)
    aupr = average_precision_score(y_true_m, y_score_m)

    return total_loss / len(val_loader.dataset), iou, auroc, aupr


def train():
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("trainval", f"../data/{config['dataset']}",
                                            config["batch_size"], config['num_workers'], augment_train=False, flipped=(config['backbone'] == 'lss'))

    class_proportions = {
        "nuscenes": [0.0206, 0.173, 0.0294, 0.777],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device)

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()

    if "pretrained" in config:
        print(f"Loading pretrained weights from {config['pretrained']}")
        model.load_state_dict(torch.load(config["pretrained"]))

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    os.makedirs(config['logdir'], exist_ok=True)

    print("--------------------------------------------------")
    print(f"Starting training on {config['type']} model with {config['backbone']} backbone")
    print(f"Using GPUS: {config['gpus']}")
    print("Training using CARLA")
    print(f"TRAIN LOADER: {len(train_loader.dataset)}")
    print(f"VAL LOADER: {len(val_loader.dataset)}")
    print(f"BATCH SIZE: {config['batch_size']}")
    print(f"OUTPUT DIRECTORY {config['logdir']} ")
    print("--------------------------------------------------")

    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(logdir=config['logdir'])

    best_iou = 0.0
    best_auroc = 0.0
    best_aupr = 0.0

    counter = 0

    for epoch in range(config['num_epochs']):
        for batchi, (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels) in enumerate(
                train_loader):
            t0 = time()
            opt.zero_grad()

            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            labels = labels.to(device)
            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)
                save_pred(preds, labels, config['logdir'])

            if counter % 50 == 0:
                intersection, union = get_iou(preds, labels)
                iou = [intersection[i] / union[i] for i in range(0, num_classes)]

                print(counter, "iou: ", iou)

                for i in range(0, num_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], counter)

                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % config['val_step'] == 0:

                model.eval()
                print("Running EVAL...")
                val_loss, val_iou, auroc, aupr = get_val(model, val_loader, device, loss_fn, activation, num_classes)
                print(f"VAL loss: {val_loss}, iou: {val_iou}, auroc {auroc}, aupr {aupr}")

                save_path = os.path.join(config['logdir'], f"model{counter}.pt")
                print(f"Saving Model: {save_path}")
                torch.save(model.state_dict(), save_path)

                if sum(val_iou) / len(val_iou) >= best_iou:
                    best_iou = sum(val_iou) / len(val_iou)
                    print(f"New best IOU model found. iou: {val_iou}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_iou.pt"))
                if auroc >= best_auroc:
                    best_auroc = auroc
                    print(f"New best AUROC model found. iou: {auroc}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_auroc.pt"))
                if aupr >= best_aupr:
                    best_aupr = aupr
                    print(f"New best AUPR model found. iou: {aupr}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_aupr.pt"))

                model.train()

                writer.add_scalar('val/loss', val_loss, counter)
                writer.add_scalar('val/auroc', auroc, counter)
                writer.add_scalar('val/aupr', aupr, counter)

                for i in range(0, num_classes):
                    writer.add_scalar(f'val/{classes[i]}_iou', val_iou[i], counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)

    args = parser.parse_args()

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]

    train()
