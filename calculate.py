from tqdm import tqdm

from datasets.nuscenes import compile_data as compile_data_nuscenes
from datasets.carla import compile_data as compile_data_carla

from datasets.carla import CarlaDataset
from sklearn.utils import class_weight

from tools.loss import *
import os
import numpy as np
import cv2
import yaml
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
from models.cvt.cross_view_transformer import *
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    with open("./configs/train_nuscenes_cvt_baseline.yaml", 'r') as file:
        config = yaml.safe_load(file)
    config['batch_size'] = 1
    config['num_workers'] = 1

    train_loader, val_loader = compile_data_nuscenes("mini", config, shuffle_train=False)
    cvt = CrossViewTransformer(outC=1)
    imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels = next(iter(train_loader))
    print(cvt.convert(intrins))
    cv2.imwrite("labels.jpg", labels[0,0].cpu().numpy()*255)
    cv2.imwrite("image.jpg", imgs[0, 0].permute(1, 2, 0).cpu().numpy() * 255)
