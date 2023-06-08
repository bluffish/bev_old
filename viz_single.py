import matplotlib.pyplot as plt
import yaml
import argparse
import seaborn as sns
from eval import eval
from tools.utils import *
from datasets.carla import compile_data

from sklearn.metrics import *
import matplotlib
from time import time

sns.set_style('white')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

class_proportions = {
    "nuscenes": [.015, .2, .05, .735],
    "carla": [0.0141, 0.3585, 0.02081, 0.6064]
}

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# matplotlib.rcParams.update({'font.size': 3})

denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),))


def graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument("config")
    args = parser.parse_args()

    is_ood = args.ood
    gt = args.gt
    num_classes = 4

    if args.gpus is None:
        device = 0
    else:
        device = int(args.gpus[0])

    gpus = [int(i) for i in args.gpus]

    indices = [1401, 1633]
    samples = len(indices)
    fig, axs = plt.subplots(samples, 3, figsize=(6, 2 * samples))

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if is_ood:
        print("USING OOD")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config['batch_size'] = 1
        config['num_workers'] = 1

    activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device, use_seg=config['seg'])
    train_loader, val_loader = compile_data("mini", config, shuffle_train=True, ood=False, cvp='val3')

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=gpus).to(device).eval()
    model.load_state_dict(torch.load(config['model_path']))

    if config['type'] == 'baseline' or config['type'] == 'dropout' or config['type'] == 'ensemble':
        al = entropy
        ep = varep
    elif config['type'] == 'enn' or config['type'] == 'postnet':
        al = aleatoric
        ep = vacuity

    if config['type'] == 'dropout':
        model.module.tests = 20
        model.module.train()

    for i, index in enumerate(indices):
        (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) = val_loader.dataset[index]

        imgs = imgs[None]
        rots = rots[None]
        trans = trans[None]
        intrins = intrins[None]
        extrins = extrins[None]
        post_rots = post_rots[None]
        post_trans = post_trans[None]
        labels = labels[None]
        ood = ood[None]

        imgs, s_labels = parse(imgs, gt)
        preds, _ = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)

        axs[i, 0].imshow(map(labels[0].permute(1, 2, 0))/255)
        # axs[i, 1].imshow(map(preds[0].permute(1, 2, 0))/255)
        axs[i, 1].imshow(plt.cm.jet(al(preds)[0][0].detach().cpu()))
        axs[i, 2].imshow(plt.cm.jet(ep(preds)[0][0].detach().cpu()))
        # preds = activation(preds).detach().cpu()

    plt.tight_layout()

    fig.savefig(f"viz_{config['backbone']}_{config['type']}.png", bbox_inches='tight', dpi=300)


graph()
