import matplotlib.pyplot as plt
import yaml

from eval import eval


def graph():
    plt.rcParams.update({'font.size': 14})

    configs = ["./configs/eval_carla_lss_baseline.yaml",
               "./configs/eval_carla_lss_enn.yaml",
               "./configs/eval_carla_lss_postnet.yaml",
               # "./configs/eval_carla_lss_ensemble.yaml",
               "./configs/eval_carla_lss_dropout.yaml"]

    paths = ["./carla/lss_baseline/best_aupr.pt",
             "./carla/lss_enn/best_aupr.pt",
             "./carla/lss_postnet/best_aupr.pt",
             # "./carla/lss_ensemble/best_iou.pt",
             "./carla/lss_dropout/best_aupr.pt"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    plt.suptitle("Misclassification")

    ar = 0

    for i in range(len(configs)):
        with open(configs[i], 'r') as file:
            config = yaml.safe_load(file)

        config['model_path'] = paths[i]

        pr, rec, fpr, tpr, aupr, auroc, iou, r = eval(config, is_ood=False, plot=False)
        ar += r

        ax1.plot(fpr, tpr, '-', label=f'{config["backbone"]}-{config["type"]} - {auroc:.3f}')
        ax2.plot(rec, pr, '-', label=f'{config["backbone"]}-{config["type"]} - {aupr:.3f}')

    ax1.set_ylim([-.05, 1.05])
    ax1.set_xlim([-.05, 1.05])
    ax2.set_ylim([-.05, 1.05])
    ax2.set_xlim([-.05, 1.05])

    no_skill = ar / len(configs)

    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
    ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=14, ncol=1)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=14, ncol=1)
    fig.savefig(f"all_combined.png", dpi=300, bbox_inches='tight')


graph()
