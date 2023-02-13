from time import time

import torch.nn as nn
from tensorboardX import SummaryWriter

from models.lift_splat_shoot import LiftSplatShoot
from models.lift_splat_shoot_gpn import LiftSplatShootGPN
from models.bevdet import BEVDet

from datasets.carla import CarlaDataset
from datasets.nuscenes import compile_data

from tools.utils import *
from tools.uncertainty import *
from tools.gpn_loss import *

import argparse
import yaml
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def get_val(model, val_loader, device, loss_fn, activation):
    model.eval()

    total_loss = 0.0

    vehicle_iou = 0.0
    background_iou = 0.0
    road_iou = 0.0
    lane_iou = 0.0

    print('running eval...')

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):
            t0 = time()

            preds = model(imgs,
                          rots,
                          trans,
                          intrins,
                          post_rots,
                          post_trans)
            labels = labels.to(device)
            loss = None

            if config['type'] == 'postnet_uce' \
                    or config['type'] == 'postnet_uce_cnn' \
                    or config['type'] == 'baseline_uce':
                preds = activation(preds)
                loss = loss_fn(preds, labels)
            elif config['type'] == 'baseline_ce':
                loss = loss_fn(preds, labels)
                preds = activation(preds, dim=1)
            elif config['type'] == 'postnet_ce':
                preds = activation(preds)
                loss = loss_fn(preds.log(), torch.argmax(labels, dim=1))

            total_loss += loss * preds.shape[0]

            intersection, union = get_iou(preds, labels)

            if union[0] == 0:
                vehicle_iou += 1.0
            else:
                vehicle_iou += (intersection[0] / union[0]) * preds.shape[0]

            if union[1] == 0:
                road_iou += 1.0
            else:
                road_iou += (intersection[1] / union[1]) * preds.shape[0]

            if union[2] == 0:
                lane_iou += 1.0
            else:
                lane_iou += (intersection[2] / union[2]) * preds.shape[0]

            if union[3] == 0:
                background_iou += 1.0
            else:
                background_iou += (intersection[3] / union[3]) * preds.shape[0]

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'vehicle_iou': vehicle_iou / len(val_loader.dataset),
        'road_iou': road_iou / len(val_loader.dataset),
        'lane_iou': lane_iou / len(val_loader.dataset),
        'background_iou': background_iou / len(val_loader.dataset),
    }


def train():
    if config['dataset'] == 'carla':
        train_dataset = CarlaDataset(os.path.join("../data/carla/", "train/"))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'],
                                                   drop_last=True)

        val_dataset = CarlaDataset(os.path.join("../data/carla/", "val/"))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config['batch_size'],
                                                 shuffle=True,
                                                 num_workers=config['num_workers'],
                                                 drop_last=True)
    elif config['dataset'] == 'nuscenes':
        train_loader, val_loader = compile_data("trainval", "../data/nuscenes", bsz=config["batch_size"], nworkers=config['num_workers'])

    gpus = config['gpus']
    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    num_classes = 4

    if config['type'] == 'baseline_ce':
        activation = torch.softmax
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'baseline_uce':
        activation = activate_uncertainty
        loss_fn = uce_loss
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'postnet_ce':
        activation = activate_gpn
        loss_fn = torch.nn.NLLLoss(weight=torch.tensor([2.0, 1.0, 4.0, 1.0])).cuda(device)
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif config['type'] == 'postnet_uce':
        activation = activate_gpn
        loss_fn = uce_loss
        model = LiftSplatShootGPN(outC=num_classes)
        model.bevencode.last = None
    elif config['type'] == 'postnet_uce_cnn':
        activation = activate_gpn
        loss_fn = uce_loss
        model = LiftSplatShootGPN(outC=num_classes)
        print(f"LATENT_SIZE: {model.bevencode.latent_size}")
    else:
        raise ValueError("Please pick a valid model type.")

    model = nn.DataParallel(model, device_ids=gpus).to(device).train()

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print("--------------------------------------------------")
    print(f"Starting training on {config['type']} model")
    print(f"Using GPUS: {gpus}")
    print("Training using CARLA")
    print("TRAIN LOADER: ", len(train_loader.dataset))
    print("VAL LOADER: ", len(val_loader.dataset))
    print("BATCH SIZE: ", config["batch_size"])
    print("--------------------------------------------------")

    counter = 0

    torch.autograd.set_detect_anomaly(True)

    out_path = f"./{config['logdir']}/{config['type']}"
    writer = SummaryWriter(logdir=out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for epoch in range(config['num_epochs']):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, labels) in enumerate(
                train_loader):
            t0 = time()
            opt.zero_grad()

            preds = model(imgs,
                          rots,
                          trans,
                          intrins,
                          post_rots,
                          post_trans)
            labels = labels.to(device)
            loss = None

            if config['type'] == 'postnet_uce' \
                    or config['type'] == 'postnet_uce_cnn' \
                    or config['type'] == 'baseline_uce':
                preds = activation(preds)
                loss = loss_fn(preds, labels)
            elif config['type'] == 'baseline_ce':
                loss = loss_fn(preds, labels)
                preds = activation(preds, dim=1)
            elif config['type'] == 'postnet_ce':
                preds = activation(preds)
                loss = loss_fn(preds.log(), torch.argmax(labels, dim=1))

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)
                save_pred(preds, labels, out_path)

            if counter % 50 == 0:
                intersection, union = get_iou(preds, labels)

                print(counter, "iou:", [intersection[0] / union[0],
                                        intersection[1] / union[1],
                                        intersection[2] / union[2],
                                        intersection[3] / union[3]])

                writer.add_scalar('train/vehicle_iou', intersection[0] / union[0], counter)
                writer.add_scalar('train/road_iou', intersection[1] / union[1], counter)
                writer.add_scalar('train/lane_iou', intersection[2] / union[2], counter)
                writer.add_scalar('train/background_iou', intersection[3] / union[3], counter)

                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % config['val_step'] == 0:
                val_info = get_val(model, val_loader, device, loss_fn, activation)
                print('VAL', val_info)

                model.eval()
                mname = os.path.join(out_path, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/vehicle_iou', val_info['vehicle_iou'], counter)
                writer.add_scalar('val/road_iou', val_info['road_iou'], counter)
                writer.add_scalar('val/lane_iou', val_info['lane_iou'], counter)
                writer.add_scalar('val/background_iou', val_info['background_iou'], counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train()
