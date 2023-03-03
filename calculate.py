from tqdm import tqdm

from datasets.nuscenes import compile_data
from datasets.carla import CarlaDataset

from tools.gpn_loss import *
import os

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    train_dataset = CarlaDataset(os.path.join("../data/carla/", "train/"))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=True)
    with torch.no_grad():
        for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(train_loader):
            pass
