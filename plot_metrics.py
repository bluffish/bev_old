import argparse, yaml

import torch

from eval import eval

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.25,
                rc={"lines.linewidth": 2.5})


def graph():

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    evidence = []
    models = range(1000, 15001, 1000)

    for i in models:
        config['model_path'] = f"{model_path}/model{i}.pt"
        print(config)
        predictions, ground_truths, uncertainty_scores, uncertainty_labels, iou = eval(config)

        evidence.append(torch.mean(predictions))

    ax1.plot([i/1000 for i in models], evidence)
    ax1.set_xticks([i/1000 for i in models])
    ax1.set_xlabel("1000 steps")
    ax1.set_ylabel("Average Evidence")

    plt.tight_layout()

    fig.savefig("evidence.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-p', '--model_path', required=False)
    parser.add_argument('-l', '--logdir', required=False)

    args = parser.parse_args()

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)
    if args.model_path is not None:
        model_path = args.model_path
    if args.logdir is not None:
        config['logdir'] = args.logdir

    graph()