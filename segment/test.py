import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
sys.path.append("..")
from datasets.nuscenes import compile_data as compile_data_nuscenes, denormalize_img

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    mask = mask['segmentation']
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

train_loader, val_loader = compile_data_nuscenes("mini", f"../../data/nuscenes",
                                        1,
                                        64)


count = 0
for (imgs, rots, trans, intrins, post_rots, post_trans, labels) in tqdm(val_loader):
    image = np.array(denormalize_img(imgs[0, 0]))

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.savefig(f"image{count}.jpg")
    stime = time.time()
    masks = mask_generator.generate(image)

    print(masks)

    etime = time.time()

    print(etime-stime)

    for i, mask in enumerate(masks):
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(f"mask{count}.jpg")
    count = count + 1
    plt.close()
