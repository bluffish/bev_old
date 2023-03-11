
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.utils import *
from tools.uncertainty import *
from tools.loss import *

import argparse
import yaml
from sklearn.manifold import TSNE
# from openTSNE import TSNE
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

torch.multiprocessing.set_sharing_strategy('file_system')


def scatter(x, classes, colors):
    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                          data=np.column_stack((x,
                                                colors)))
    cps_df.loc[:, 'target'] = cps_df.target.astype(int)
    cps_df.head()
    grid = sns.FacetGrid(cps_df, hue="target", height=10, legend_out=False)
    plot = grid.map(plt.scatter, 'CP1', 'CP2')

    plot.add_legend()

    for t, l in zip(plot._legend.texts, classes):
        t.set_text(l)

    return plot


def eval():
    if config['dataset'] == 'carla':
        train_loader, val_loader = compile_data_carla("../data/carla", config["batch_size"],
                                                      config['num_workers'])
    elif config['dataset'] == 'nuscenes':
        train_loader, val_loader = compile_data_nuscenes("trainval", "../data/nuscenes", config["batch_size"],
                                                         config['num_workers'])
    else:
        raise ValueError("Please pick a valid dataset.")

    gpus = config['gpus']
    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    num_classes = 4
    classes = ["vehicle", "road", "lane", "background"]

    class_proportions = {
        "nuscenes": [0.0206, 0.173, 0.0294, 0.777],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], num_classes)

    if "ce" in config['type']:
        uncertainty_function = entropy
    else:
        uncertainty_function = dissonance

    if "postnet" in config['type']:
        model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=gpus).to(device).eval()
    model.load_state_dict(torch.load(config['model_path']))

    print("--------------------------------------------------")
    print(f"Starting eval on {config['type']} model")
    print(f"Using GPUS: {gpus}")
    print("Training using CARLA ")
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    print('Running eval...')

    iou = [0.0] * num_classes

    y_true = []
    y_scores = []

    if len(Path(config['logdir']).parents) == 1:
        out_path = f"./{config['logdir']}/{config['type']}"
    else:
        out_path = config['logdir']

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if config['tsne']:
        print("Running TSNE...")

        tsne = TSNE(n_components=2)

        model.module.bevencode.tsne = True

        tsne_path = os.path.join(out_path, './tsne')
        if not os.path.exists(tsne_path):
            os.makedirs(tsne_path)

        for i in range(10):
            imgs, rots, trans, intrins, post_rots, post_trans, labels = next(iter(val_loader))
            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
            labels = labels.cpu()

            embedding = tsne.fit(preds.view(preds.shape[0]*preds.shape[1], 40000).transpose(0, 1).detach().cpu().numpy())
            f = scatter(embedding, classes, torch.argmax(labels.view(4, 40000), dim=0).cpu().numpy())
            f.savefig(os.path.join(tsne_path, f"{config['type']}_{i}.png"))

        model.module.bevencode.tsne = False

    print("Done!")

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):

            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)

            labels = labels.to(device)
            uncertainty = uncertainty_function(preds)

            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            intersect, union = get_iou(preds, labels)

            for i in range(0, num_classes):
                if union[0] == 0:
                    iou[i] += 1.0
                else:
                    iou[i] += intersect[i] / union[i] * preds.shape[0]

            save_pred(preds, labels, out_path)
            plt.imsave(os.path.join(out_path, "uncertainty_map.jpg"), plt.cm.jet(uncertainty[0][0]))

            preds = preds[:, 0, :, :].ravel()
            labels = labels[:, 0, :, :].ravel()
            uncertainty = torch.tensor(uncertainty).ravel()

            vehicle = np.logical_or(preds.cpu() > 0.5, labels.cpu() == 1).bool()

            preds = preds[vehicle]
            labels = labels[vehicle]
            uncertainty = uncertainty[vehicle]

            pred = (preds > 0.5)
            tgt = labels.bool()
            intersect = (pred == tgt).type(torch.int64)

            # y_true += intersect.tolist()
            y_true += labels.cpu().tolist()
            uncertainty = -uncertainty
            y_scores += uncertainty.tolist()

    print(f'iou: {iou}')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    eval(config)
