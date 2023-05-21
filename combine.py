import matplotlib.pyplot as plt
import yaml

from eval import eval


def graph():
    plt.rcParams.update({'font.size': 14})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    configs = ["./configs/eval_carla_lss_baseline.yaml",
               "./configs/eval_carla_lss_enn.yaml",
               "./configs/eval_carla_lss_postnet.yaml",
               "./configs/eval_carla_lss_ensemble.yaml",
               "./configs/eval_carla_lss_dropout.yaml"]

    # paths = ["./carla/lss_baseline/model7000.pt",
    #          "./carla/lss_enn/model1000.pt",
    #          "./carla/lss_postnet/model1000.pt",
    #          "./carla/lss_ensemble/model9000.pt",
    #          "./carla/lss_dropout/model3000.pt",
    #          "./carla/lss_dropout/model3000.pt"]

    paths = ["./carla/lss_baseline/best_auroc.pt",
             "./carla/lss_enn/best_auroc.pt",
             "./carla/lss_postnet/best_auroc.pt",
             "./carla/lss_ensemble/best_auroc.pt",
             "./carla/lss_dropout/best_auroc.pt",
             "./carla/lss_dropout/best_auroc.pt"]


    for i in range(len(configs)):
        with open(configs[i], 'r') as file:
            config = yaml.safe_load(file)

        config['model_path'] = paths[i]

        pr, rec, fpr, tpr, aupr, auroc, iou = eval(config)

        ax1.plot(fpr, tpr, label=f"{config['backbone']}_{config['type']}\nAUROC={auroc:.3f}")
        ax2.plot(rec, pr, label=f"{config['backbone']}_{config['type']}\nAUPR={aupr:.3f}")

    ax1.set_ylim([-.05, 1.05])
    ax1.set_xlim([-.05, 1.05])
    ax2.set_ylim([-.05, 1.05])
    ax2.set_xlim([-.05, 1.05])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=14, ncol=1)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=14, ncol=1)
    fig.savefig(f"all_combined.png", dpi=300, bbox_inches='tight')


graph()
