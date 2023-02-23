from tqdm import tqdm

from datasets.nuscenes import compile_data
from tools.gpn_loss import *


import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    train_loader, val_loader = compile_data("trainval", "../data/nuscenes/", 512, 1)

    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):
            pass
