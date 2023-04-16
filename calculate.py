from tqdm import tqdm

from datasets.nuscenes import compile_data as compile_data_nuscenes
from datasets.carla import compile_data as compile_data_carla

from datasets.carla import CarlaDataset
from sklearn.utils import class_weight

from tools.loss import *
import os
import numpy as np
import cv2

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    train_loader, val_loader = compile_data_nuscenes("mini", f"../data/nuscenes", 1,
                                            1, ood=False, shuffle_train=False)

    imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels = next(iter(train_loader))

    cv2.imwrite("labels.jpg", labels[0,0].cpu().numpy()*255)
