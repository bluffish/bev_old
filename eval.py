import torch

from datasets.nuscenes import compile_data as compile_data_nuscenes
from datasets.carla import compile_data as compile_data_carla
from sklearn.manifold import TSNE
from sklearn.metrics import *
from tqdm import tqdm

from tools.utils import *
from tools.uncertainty import *
from tools.loss import *

import argparse
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def eval(config, plot=True, is_ood=False):
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("mini" if is_ood else "mini", config, shuffle_train=True, ood=is_ood)

    class_proportions = {
        "nuscenes": [.015, .2, .05, .735],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device)

    if config['type'] == 'baseline' or config['type'] == 'dropout' or config['type'] == 'ensemble':
        uncertainty_function = entropy
    elif config['type'] == 'enn' or config['type'] == 'postnet':
        if is_ood:
            uncertainty_function = vacuity
        else:
            uncertainty_function = dissonance

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).eval()
    model.load_state_dict(torch.load(config['model_path']))

    if config['type'] == "dropout":
        model.module.tests = 20
        model.module.train()

    print("--------------------------------------------------")
    print(f"Starting eval on {config['type']} model")
    print(f"Using GPUS: {config['gpus']}")
    print(f"BATCH SIZE: {config['batch_size']}")
    print(f"Eval using {config['dataset']} ")
    print("VAL LOADER: ", len(val_loader.dataset))
    print(f"OUTPUT DIRECTORY {config['logdir']} ")
    print(f"OOD: {is_ood}")
    print(f"MODEL PATH: {config['model_path']}")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    if config['tsne']:
        print("Running TSNE...")

        tsne = TSNE(n_components=2)

        model.module.bevencode.tsne = True

        tsne_path = os.path.join(config['logdir'], 'tsne')
        os.makedirs(tsne_path, exist_ok=True)

        imgs, rots, trans, intrins, post_rots, post_trans, labels = next(iter(val_loader))
        preds = model(imgs, rots, trans, intrins, post_rots, post_trans).detach().cpu()

        for i in range(config['batch_size']):
            l = torch.argmax(labels[i].view(num_classes, -1), dim=0).cpu().numpy()
            feature_map = tsne.fit_transform(preds[i].view(num_classes, -1).transpose(0, 1))

            f = scatter(feature_map, classes, l)
            print(f"Saving TSNE plot at {os.path.join(tsne_path, str(i))}")
            plt.savefig(os.path.join(tsne_path, str(i)))

        model.module.bevencode.tsne = False

        print("Done!")

    iou = [0.0] * num_classes

    y_true = []
    y_score = []

    with torch.no_grad():
        for (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in tqdm(val_loader):

            preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            uncertainty = uncertainty_function(preds).cpu()

            try:
                preds = activation(preds)
            except Exception:
                preds = activation(preds, dim=1)

            labels = labels.to(device)

            cv2.imwrite(os.path.join(config['logdir'], "uncertainty_map.jpg"),
                       cv2.cvtColor((plt.cm.jet(uncertainty[0][0])*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            save_pred(preds, labels, config['logdir'])

            if is_ood:
                cv2.imwrite(os.path.join(config['logdir'], f"ood.jpg"),
                           ood[0].cpu().numpy()*255)

                y_true += ood.ravel().cpu()
                y_score += uncertainty.ravel().cpu()
            else:
                intersect, union = get_iou(preds, labels)

                for cl in range(0, num_classes):
                    iou[cl] += 1 if union[0] == 0 else intersect[cl] / union[cl] * preds.shape[0]

                pmax = torch.argmax(preds, dim=1).cpu()
                lmax = torch.argmax(labels, dim=1).cpu()

                u = uncertainty.ravel()

                misclassified = pmax != lmax
                cv2.imwrite(os.path.join(config['logdir'], "mask.jpg"), misclassified[0].cpu().numpy()*255)

                y_true += misclassified.ravel()
                y_score += u

    if is_ood:
        pr, rec, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        y_score_binary = [x > .5 for x in y_score]
        print(classification_report(y_true, y_score_binary))
        plt.ylim([0, 1.05])

        aupr = average_precision_score(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        rcd = RocCurveDisplay(fpr=fpr, tpr=tpr)
        prd = PrecisionRecallDisplay(precision=pr, recall=rec)
        rcd.plot(ax=ax1, label=f"OOD\nAUROC={auroc:.3f}")
        prd.plot(ax=ax2, label=f"OOD\nAUPR={aupr:.3f}")
        plt.scatter(rec, pr, c="red")

        ax1.legend()
        ax2.legend()

        plt.ylim([0, 1.05])
        fig.suptitle(f"OOD - {config['backbone']}-{config['type']}")

        save_path = os.path.join(config['logdir'], f"combined_ood.jpg")
        print(f"Saving combined for OOD at {save_path}\n"
              f"OOD - AUPR: {aupr} AUROC: {auroc}")
        plt.savefig(save_path)
        plt.clf()
        return pr, rec, fpr, tpr, aupr, auroc

    else:
        iou = [i / (len(val_loader.dataset) - len(val_loader.dataset) % config['batch_size']) for i in iou]

        print(f'iou: {iou}')

        pr, rec, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        aupr = auc(rec, pr)
        auroc = auc(fpr, tpr)
        no_skill = np.sum(y_true) / len(y_true)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.plot(fpr, tpr, 'b-', label=f'AUROC - {auroc:.3f}')
            ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.legend()

            ax2.plot(rec, pr, 'r-', label=f'AUPR - {aupr:.3f}')
            ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.legend()

            plt.ylim([0, 1.05])
            fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {config['backbone']}-{config['type']}")

            save_path = os.path.join(config['logdir'], f"combined_{'ood' if is_ood else 'misclassification'}_{config['type']}.jpg")
            print(f"Saving combined {'ood' if is_ood else 'misclassification'} at {save_path}\n"
                  f"AUPR: {aupr} AUROC: {auroc}")

            plt.savefig(save_path)

        return pr, rec, fpr, tpr, aupr, auroc, iou, no_skill


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-p', '--model_path', required=False)
    parser.add_argument('-l', '--logdir', required=False)

    args = parser.parse_args()

    is_ood = False

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.ood is not None:
        is_ood = args.ood
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)
    if args.model_path is not None:
        config['model_path'] = args.model_path
    if args.logdir is not None:
        config['logdir'] = args.logdir

    eval(config, is_ood=is_ood)