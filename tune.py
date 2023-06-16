from tensorboardX import SummaryWriter

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.utils import *
import torch
import torch.nn as nn
import argparse
import yaml
from eval import get
import random
import warnings
from time import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
# torch.backends.cudnn.enabled = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def train():
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("trainval", config, shuffle_train=True, ood=True)

    class_proportions = {
        "nuscenes": [.015, .2, .05, .735],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation = activate_uce
    loss_fn = UCELossReg(weights=torch.tensor([3.0, 1.0, 2.0, 1.0]).to(device)).to(device)
    model = LiftSplatShootENN(outC=4)

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()

    uncertainty_function = vacuity

    if pretrained is not None:
        print(f"Loading pretrained weights from {pretrained}")
        model.load_state_dict(torch.load(pretrained), strict=False)

    if config['backbone'] == 'lss':
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = None
        print("Using Adam")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, div_factor=10, pct_start=.3, final_div_factor=10,
                                                        max_lr=config['learning_rate'], total_steps=config['num_steps'])
        print("Using AdamW and OneCycleLR")

    os.makedirs(config['logdir'], exist_ok=True)

    print("--------------------------------------------------")
    print(f"Starting training on {config['type']} model with {config['backbone']} backbone")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print("--------------------------------------------------")

    config['val_step'] = 250

    writer = SummaryWriter(logdir=config['logdir'])

    best_iou, step, epoch = 0, 0, 1

    torch.autograd.set_detect_anomaly(True)
    while True:
        for batchi, (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in enumerate(
                train_loader):
            t0 = time()
            opt.zero_grad(set_to_none=True)
            labels = labels.to(device)
            ood = ood.to(device)

            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            loss = loss_fn(preds, labels, ood)
            uncertainty = uncertainty_function(preds).detach().cpu()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            step += 1
            t1 = time()

            preds = activation(preds)

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(step, loss.item())

                writer.add_scalar('train/step_time', t1 - t0, step)
                writer.add_scalar('train/loss', loss, step)
                save_pred(preds, labels, config['logdir'])
                cv2.imwrite(os.path.join(config['logdir'], "uncertainty_map.png"),
                            cv2.cvtColor((plt.cm.jet(uncertainty[0][0]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(config['logdir'], f"ood.png"),
                            ood[0].cpu().numpy() * 255)

            if step % 50 == 0:
                intersection, union = get_iou(preds, labels)
                iou = [intersection[i] / union[i] for i in range(0, num_classes)]

                print(step, "iou: ", iou)

                for i in range(0, num_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

                writer.add_scalar('train/epoch', epoch, step)

            if step % config['val_step'] == 0:
                model.eval()
                print("Running EVAL...")

                predictions, ground_truths, uncertainty_scores, uncertainty_labels = get(model, val_loader,
                                                                                         uncertainty_function,
                                                                                         activation,
                                                                                         device,
                                                                                         config, is_ood=True)

                intersect, union = get_iou(torch.softmax(predictions, dim=1), ground_truths)
                val_iou = [intersect[i] / union[i] for i in range(len(intersect))]

                fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)
                pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=torch.mean(uncertainty_scores))

                print(f"mIOU: {val_iou}")
                print(f"AUROC {auroc:.3f}")
                print(f"AUPR {aupr:.3f}",)
                print(f"PAvPU {pavpu:.3f}")
                print(f"p(accurate|certain) {agc:.3f}")
                print(f"p(uncertain|inaccurate) {ugi:.3f}")

                save_path = os.path.join(config['logdir'], f"model{step}.pt")
                print(f"Saving Model: {save_path}")
                torch.save(model.state_dict(), save_path)

                if sum(val_iou) / len(val_iou) >= best_iou:
                    best_iou = sum(val_iou) / len(val_iou)
                    print(f"New best IOU model found. iou: {val_iou}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_iou.pt"))

                model.train()

                writer.add_scalar('val/AUROC', auroc, step)
                writer.add_scalar('val/AUPR', aupr, step)
                writer.add_scalar('val/PAvPU', pavpu, step)
                writer.add_scalar('val/p_accurate_certain_', agc, step)
                writer.add_scalar('val/p_uncertain_inaccurate_', ugi, step)

                for i in range(0, num_classes):
                    writer.add_scalar(f'val/{classes[i]}_iou', val_iou[i], step)

            if step == config['num_steps']:
                return

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-l', '--logdir', required=False)
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-p', '--pretrained', required=False)

    args = parser.parse_args()
    pretrained = args.pretrained

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.logdir is not None:
        config['logdir'] = args.logdir
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)

    train()
