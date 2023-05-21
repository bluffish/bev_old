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
import random

torch.backends.cudnn.enabled = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.multiprocessing.set_sharing_strategy('file_system')

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def get_val(model, val_loader, device, loss_fn, activation, num_classes):
    total_loss = 0.0
    iou = [0.0] * num_classes

    y_true_m = []
    y_score_m = []
    c = 0

    with torch.no_grad():
        for (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in tqdm(val_loader):
            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            uncertainty = vacuity(preds).cpu()
            labels = labels.to(device)
            ood = ood.to(device)

            loss = loss_fn(preds, labels, ood)
            preds = activation(preds)

            total_loss += loss * preds.shape[0]
            intersection, union = get_iou(preds, labels)

            for i in range(0, num_classes):
                if union[0] == 0:
                    iou[i] += 1.0
                else:
                    iou[i] += intersection[i] / union[i] * preds.shape[0]

            if c < 100:
                save_pred(preds, labels, config['logdir'])

                mask = np.logical_or(uncertainty[:, 0, :, :].cpu() > .5, ood.cpu() == 1).bool()
                l = ood[mask].ravel()
                u = uncertainty[:, 0, :, :][mask].ravel()

                plt.imsave(os.path.join(config['logdir'], "uncertainty_map.jpg"),
                           plt.cm.jet(uncertainty[0][0].cpu()))
                cv2.imwrite(os.path.join(config['logdir'], "ood.jpg"),
                            ood[0].cpu().numpy() * 255)

                y_true_m += l.cpu()
                y_score_m += u.cpu()

    iou = [i / len(val_loader.dataset) for i in iou]

    try:
        auroc = roc_auc_score(y_true_m, y_score_m)
        aupr = average_precision_score(y_true_m, y_score_m)
    except:
        auroc = 0
        aupr = 0

    return total_loss / len(val_loader.dataset), iou, auroc, aupr


def train():
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]
    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("mini", config, shuffle_train=True, ood=True)

    class_proportions = {
        "nuscenes": [.015, .2, .05, .735],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation = activate_uce
    loss_fn = UCELossReg(weights=torch.tensor([3.0, 1.0, 2.0, 1.0])).to(device)
    model = CrossViewTransformerENN(outC=4)

    model.p_c = torch.tensor(class_proportions[config['dataset']])
    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()

    if "pretrained" in config:
        print(f"Loading pretrained weights from {config['pretrained']}")
        model.load_state_dict(torch.load(config["pretrained"]))

    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print("Using AdamW and OneCycleLR")

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

    writer = SummaryWriter(logdir=config['logdir'])
    uncertainty_function = vacuity

    best_iou = 0.0
    best_auroc = 0.0
    best_aupr = 0.0

    step = 0
    epoch = 0

    while True:
        for batchi, (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in enumerate(
                val_loader):
            t0 = time()
            opt.zero_grad(set_to_none=True)

            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            uncertainty = uncertainty_function(preds).cpu()
            labels = labels.to(device)
            ood = ood.to(device)

            loss = loss_fn(preds, labels, ood)
            preds = activation(preds)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            step += 1
            t1 = time()

            plt.imsave(os.path.join(config['logdir'], "uncertainty_map.jpg"),
                       plt.cm.jet(uncertainty[0][0].detach().cpu().numpy()))
            cv2.imwrite(os.path.join(config['logdir'], "ood.jpg"),
                        ood[0].cpu().numpy() * 255)

            if step % 10 == 0:
                print(step, loss.item())

                writer.add_scalar('train/step_time', t1 - t0, step)
                writer.add_scalar('train/loss', loss, step)
                save_pred(preds, labels, config['logdir'])

            if step % 50 == 0:
                intersection, union = get_iou(preds, labels)
                iou = [intersection[i] / union[i] for i in range(0, num_classes)]

                cv2.imwrite(os.path.join(config['logdir'], "binary_preds.jpg"),
                            preds[0, 0].detach().cpu().numpy() * 255)
                cv2.imwrite(os.path.join(config['logdir'], "binary_labels.jpg"),
                            labels[0, 0].detach().cpu().numpy() * 255)

                print(step, "iou: ", iou)

                for i in range(0, num_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

                writer.add_scalar('train/epoch', epoch, step)

            if step % config['val_step'] == 0:

                model.eval()
                print("Running EVAL...")
                val_loss, val_iou, auroc, aupr = get_val(model, val_loader, device, loss_fn, activation, num_classes)
                print(f"VAL loss: {val_loss}, iou: {val_iou}, auroc {auroc}, aupr {aupr}")

                save_path = os.path.join(config['logdir'], f"model{step}.pt")
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

                writer.add_scalar('val/loss', val_loss, step)
                writer.add_scalar('val/auroc', auroc, step)
                writer.add_scalar('val/aupr', aupr, step)

                for i in range(0, num_classes):
                    writer.add_scalar(f'val/{classes[i]}_iou', val_iou[i], step)

            if step == config['num_steps']:
                return

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-l', '--logdir', required=False)

    args = parser.parse_args()

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)
    if args.logdir is not None:
        config['logdir'] = args.logdir


    train()
