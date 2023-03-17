from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes
from sklearn.manifold import TSNE
from sklearn.metrics import *
from tqdm import tqdm

from tools.utils import *
from tools.uncertainty import *
from tools.loss import *

import argparse
import yaml

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
                          data=np.column_stack((x, colors)))
    cps_df['target'] = cps_df['target'].astype(int)
    cps_df.head()
    grid = sns.FacetGrid(cps_df, hue="target", height=10, legend_out=False)
    plot = grid.map(plt.scatter, 'CP1', 'CP2')
    plot.add_legend()
    for t, l in zip(plot._legend.texts, classes):
        t.set_text(l)

    return plot


def eval():
    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("../data/carla", config["batch_size"], config['num_workers'])

    device = torch.device(f"cuda:{config['gpus'][0]}" if config['gpus'] else 'cpu')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    class_proportions = {
        "nuscenes": [0.0206, 0.173, 0.0294, 0.777],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], num_classes)

    if "uce" in config['type']:
        uncertainty_function = dissonance
    elif "ce" in config['type']:
        uncertainty_function = entropy

    if "postnet" in config['type']:
        model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()
    model.load_state_dict(torch.load(config['model_path']))

    if config['type'] == "dropout_ce":
        model.module.tests = 20
        model.module.train()

    print("--------------------------------------------------")
    print(f"Starting eval on {config['type']} model")
    print(f"Using GPUS: {config['gpus']}")
    print("Training using CARLA ")
    print("VAL LOADER: ", len(val_loader.dataset))
    print(f"OUTPUT DIRECTORY {config['logdir']} ")
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

    y_true = [[], [], [], []]
    y_scores = [[], [], [], []]

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):

            preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
            uncertainty = uncertainty_function(preds)

            labels = labels.to(device)

            preds, loss = get_step(preds, labels, activation, loss_fn, config['type'])

            intersect, union = get_iou(preds, labels)

            for j in range(0, num_classes):
                if union[0] == 0:
                    iou[j] += 1.0
                else:
                    iou[j] += intersect[j] / union[j] * preds.shape[0]

            save_pred(preds, labels, config['logdir'])
            plt.imsave(os.path.join(config['logdir'], "uncertainty_map.jpg"),
                       plt.cm.jet(uncertainty[0][0]))

            for j in range(num_classes):
                mask = np.logical_or(preds[:, j, :, :].cpu() > 0.5, labels[:, j, :, :].cpu() == 1).bool()

                p = preds[:, j, :, :][mask].ravel()
                l = labels[:, j, :, :][mask].ravel()
                u = torch.tensor(uncertainty[:, 0, :, :][mask]).ravel()

                p = (p > 0.5)
                tgt = l.bool()
                intersect = (p == tgt).type(torch.int64)

                y_true[j] += intersect.tolist()
                u = -u
                y_scores[j] += u.tolist()

    iou = [i / len(val_loader.dataset) for i in iou]

    print(f'iou: {iou}')

    plt.ylim([0, 1.05])

    for j in range(num_classes):
        pr, rec, _ = precision_recall_curve(y_true[j], y_scores[j])
        aupr = average_precision_score(y_true[j], y_scores[j])

        fpr, tpr, _ = roc_curve(y_true[j], y_scores[j])
        auroc = roc_auc_score(y_true[j], y_scores[j])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1, label=f"{config['type']}\nAUROC={auroc:.3f}")
        PrecisionRecallDisplay(precision=pr, recall=rec).plot(ax=ax2, label=f"{config['type']}\nAUPR={aupr:.3f}")

        ax1.legend()
        ax2.legend()

        fig.suptitle(classes[j])
        save_path = os.path.join(config['logdir'], f"combined_{classes[j]}.jpg")
        print(f"Saving combined for {classes[j]} class at {save_path}")
        plt.savefig(save_path)
        print(f"{classes[j]} CLASS - AUPR: {aupr} AUROC: {auroc}")


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

    eval()
