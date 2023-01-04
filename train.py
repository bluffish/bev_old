from time import time

import torch.nn as nn
from tensorboardX import SummaryWriter

from models.lift_splat_shoot import LiftSplatShoot
from models.lift_splat_shoot_gpn import LiftSplatShootGPN

from datasets.carla import CarlaDataset
from datasets.nuscenes import compile_data

from tools.utils import *
from tools.uncertainty import *
from tools.gpn_loss import *
from torchviz import make_dot

import argparse
import yaml
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def get_val(model, val_loader, device, loss_fn, activation, num_classes):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0

    print('running eval...')

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):

            preds = model(imgs,
                          rots,
                          trans,
                          intrins,
                          post_rots,
                          post_trans)
            labels = labels.to(device)

            if config['type'] == 'gpn':
                preds = activation(preds)
                # loss = loss_fn(preds, labels)
                loss = loss_fn(preds.log(), torch.argmax(labels, dim=1))
            else:
                try:
                    loss = loss_fn(preds, labels)
                    preds = activation(preds, dim=1)
                except Exception as e:
                    loss = loss_fn(preds.view(-1, num_classes), labels.view(-1, num_classes), 0, num_classes, 10,
                                   device)
                    preds = activation(preds)

            total_loss += loss

            try:
                preds = activation(preds)
            except Exception as e:
                preds = activation(preds, dim=1)

            intersection, union = get_iou(preds, labels)
            if union[0] == 0:
                iou = 1.0
            else:
                iou = (intersection[0] / union[0]) * preds.shape[0]

            total_iou += iou

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'iou': total_iou / len(val_loader.dataset),
    }


def train(
        config
):
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

    # train_loader, val_loader = compile_data("trainval", "../data/nuscenes/", config['batch_size'], config['num_workers'])

    gpus = config['gpus']
    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    num_classes = 3

    if config['type'] == 'entropy':
        activation = torch.softmax
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 1.0])).cuda(device)
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'dissonance' or config['type'] == 'vacuity':
        activation = activate_uncertainty
        loss_fn = edl_digamma_loss
        model = LiftSplatShoot(outC=num_classes)
    elif config['type'] == 'gpn':
        activation = activate_gpn
        loss_fn = torch.nn.NLLLoss(weight=torch.tensor([2.0, 1.0, 1.0])).cuda(device)
        # loss_fn = gpn_loss
        model = LiftSplatShootGPN(outC=num_classes)
    else:
        raise ValueError("Please pick a valid model type.")

    model = nn.DataParallel(model, device_ids=gpus).to(device).train()

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print("--------------------------------------------------")
    print(f"Starting eval on {config['type']} model")
    print(f"Using GPUS: {gpus}")
    print("Training using CARLA")
    print("TRAIN LOADER: ", len(train_loader.dataset))
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    counter = 0

    torch.autograd.set_detect_anomaly(True)

    out_path = "./outputs/"+config['name']
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

            # make_dot(preds, params=dict(model.module.bevencode.named_parameters())).render("train_torchviz", format="png")

            labels = labels.to(device)

            if config['type'] == 'gpn':
                preds = activation(preds)
                loss = loss_fn(preds, labels)
                # loss = loss_fn(preds.log(), torch.argmax(labels, dim=1))
            else:
                try:
                    loss = loss_fn(preds, labels)
                    preds = activation(preds, dim=1)
                except Exception as e:
                    loss = loss_fn(preds.view(-1, num_classes), labels.view(-1, num_classes), epoch, num_classes, 10,
                                   device)
                    preds = activation(preds)

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            counter += 1
            t1 = time()

            save_pred(preds, labels, out_path)

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                intersection, union = get_iou(preds, labels)

                print(counter, "iou:", [intersection[0] / union[0], intersection[1] / union[1], intersection[2] / union[2]])
                writer.add_scalar('train/vehicle_iou', intersection[0] / union[0], counter)
                writer.add_scalar('train/road_iou', intersection[1] / union[1], counter)
                writer.add_scalar('train/background_iou', intersection[2] / union[2], counter)

                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % config['val_step'] == 0:
                val_info = get_val(model, val_loader, device, loss_fn, activation, num_classes)
                print('VAL', val_info)

                model.eval()
                mname = os.path.join(out_path, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train(config)
