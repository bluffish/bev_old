import yaml
import argparse
import os

import matplotlib.pyplot as plt

from eval import eval

classes = ["vehicle", "road", "lane", "background"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-p', '--model_path', required=True)
    parser.add_argument('-s', '--steps', required=False)
    parser.add_argument('-c', '--cl', required=False)
    parser.add_argument('-l', '--logdir', required=True)

    args = parser.parse_args()

    is_ood = False

    os.makedirs(args.logdir, exist_ok=True)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.steps is not None:
        steps = int(args.steps)
    else: steps = 1000

    if args.cl is not None:
        c = int(args.cl)
    else:
        c = 0

    print(f"Using config {args.config}, with class {c}")

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.ood is not None:
        is_ood = args.ood

    steps_values = range(1000, steps+1, 1000)
    aupr_values = [[], [], [], []]
    auroc_values = [[], [], [], []]
    iou_values = [[], [], [], []]

    for i in steps_values:
        config['model_path'] = os.path.join(args.model_path, f"model{i}.pt")
        print(config['model_path'])
        aupr, auroc, iou = eval(config, metrics=True, is_ood=is_ood)

        for j in range(4):
            aupr_values[j].append(aupr[j])
            auroc_values[j].append(auroc[j])
            iou_values[j].append(iou[j])

    for i in range(4):
        plt.clf()
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(steps_values, aupr_values[i], label='AUPR')
        plt.xlabel('Steps')
        plt.ylabel('AUPR')
        plt.title('AUPR Plot')
        plt.xticks(steps_values)

        plt.subplot(1, 3, 2)
        plt.plot(steps_values, auroc_values[i], label='AUROC')
        plt.xlabel('Steps')
        plt.ylabel('AUROC')
        plt.title('AUROC Plot')
        plt.xticks(steps_values)

        plt.subplot(1, 3, 3)
        plt.plot(steps_values, iou_values[i], label='IOU')
        plt.xlabel('Steps')
        plt.ylabel('IOU')
        plt.title('IOU Plot')
        plt.xticks(steps_values)

        plt.tight_layout()
        plt.savefig(os.path.join(args.logdir, f"{classes[i]}_metrics_plot.jpg"))
