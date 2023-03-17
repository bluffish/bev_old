import os
from time import time

from tensorboardX import SummaryWriter

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.utils import *

import argparse
import yaml
from tqdm import tqdm


def get_val(model, val_loader, device, loss_fn, activation, num_classes):
    total_loss = 0.0
    iou = [0.0] * num_classes

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):
            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
            labels = labels.to(device)

            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            total_loss += loss * preds.shape[0]
            intersection, union = get_iou(preds, labels)

            for i in range(0, num_classes):
                if union[0] == 0:
                    iou[i] += 1.0
                else:
                    iou[i] += intersection[i] / union[i] * preds.shape[0]

    iou = [i / len(val_loader.dataset) for i in iou]

    return total_loss / len(val_loader.dataset), iou


def train():
    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("../data/carla", config["batch_size"], config['num_workers'])

    device = torch.device(f"cuda:{config['gpus'][0]}" if config['gpus'] else 'cpu')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    class_proportions = {
        "nuscenes": [0.0206, 0.173, 0.0294, 0.777],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], num_classes)

    if "postnet" in config['type']:
        model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()

    if "pretrained" in config:
        print(f"Loading pretrained weights from {config['pretrained']}")
        model.load_state_dict(torch.load(config["pretrained"]))

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    os.makedirs(config['logdir'], exist_ok=True)

    print("--------------------------------------------------")
    print(f"Starting training on {config['type']} model")
    print(f"Using GPUS: {config['gpus']}")
    print("Training using CARLA")
    print(f"TRAIN LOADER: {len(train_loader.dataset)}")
    print(f"VAL LOADER: {len(val_loader.dataset)}")
    print(f"BATCH SIZE: {config['batch_size']}")
    print(f"OUTPUT DIRECTORY {config['logdir']} ")
    print("--------------------------------------------------")

    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(logdir=config['logdir'])

    best = 0.0
    counter = 0

    for epoch in range(config['num_epochs']):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, labels) in enumerate(
                train_loader):
            t0 = time()
            opt.zero_grad()

            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
            labels = labels.to(device)

            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
                val_loss, val_iou = get_val(model, val_loader, device, loss_fn, activation, num_classes)
                print(f"VAL loss: {val_loss}, iou: {val_iou},  average: {best}")

                save_path = os.path.join(config['logdir'], f"model{counter}.pt")
                print(f"Saving Model: {save_path}")
                torch.save(model.state_dict(), save_path)

                if sum(val_iou) / len(val_iou) >= best:
                    best = sum(val_iou) / len(val_iou)
                    print(f"New best model found. iou: {val_iou}")
                torch.save(model.state_dict(), os.path.join(config['logdir'], "best.pt"))

                model.train()

                writer.add_scalar('val/loss', val_loss, counter)

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
