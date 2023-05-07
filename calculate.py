from tqdm import tqdm

from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.loss import *
import yaml
import seaborn as sns

from tools.utils import save_pred

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
from models.cvt.cross_view_transformer import *
torch.multiprocessing.set_sharing_strategy('file_system')

hw = [
    (56, 120),
    (14, 30),
]

H, W = 25, 25

query_locations = [
] # (Y, X)


def map (atts, cidx, x, y):
    atts = atts.cpu().numpy()

    max_x = 0
    max_y = 0
    max_att = 0

    for mx in range(25):
        for my in range(25):
            am = atts[mx][my][cidx][x][y]

            if am > max_att:
                max_att = am
                max_x = mx
                max_y = my

    return max_x, max_y, max_att

if __name__ == "__main__":
    with open("./configs/train_nuscenes_cvt_baseline.yaml", 'r') as file:
        config = yaml.safe_load(file)
    config['batch_size'] = 1
    config['num_workers'] = 1

    train_loader, val_loader = compile_data_nuscenes("mini", config, shuffle_train=False)

    cvt = nn.DataParallel(CrossViewTransformer(outC=4), device_ids=[0])
    cvt.load_state_dict(torch.load("./nuscenes/cvt_baseline_old/best_iou.pt"))

    # imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels = next(iter(train_loader))
    t = iter(train_loader)

    imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels = next(t)
    preds, atts = cvt(imgs, rots, trans, intrins, extrins, post_rots, post_trans, return_att=True)

    rearranged_atts = []
    for i, att in enumerate(atts):
        rearrangd_att = rearrange(att, '(b m) (H W) (n h w) -> b m H W n h w', n=6, m=4, H=H, W=W, h=hw[i][0],
                                  w=hw[i][1])
        rearranged_atts.append(rearrangd_att)

    # Visualization
    att = rearranged_atts[-1]  # get the last attention layer
    att = att[0].detach()  # batch size is 1, we remove the batch dim here.
    mean_att = torch.mean(att, dim=0)
    preds = preds.softmax(dim=1)
    pred, labels = save_pred(preds, labels, './')

    cam_img = np.transpose(imgs[0][1].cpu().numpy(), (1, 2, 0))
    x, y, k = map(mean_att, 1, 1, 20)

    cv2.imwrite("labels.jpg", labels)
    cv2.imwrite("fc.jpg", cam_img*255)

    for query_location in query_locations:
        att_map = mean_att[query_location[0]][query_location[1]].detach().cpu().numpy()

        pred_img = preds[0][0].detach().cpu().numpy()
        plt.title('Prediction and Query')
        plt.imshow(pred_img)
        # Query mask
        query_mask = np.zeros((H, W))
        query_mask[query_location[0], query_location[1]] = 1
        query_mask = cv2.resize(query_mask, dsize=(pred_img.shape[1], pred_img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        plt.imshow(query_mask, alpha=query_mask)
        plt.savefig("query_mask.jpg")

        ova = np.unravel_index(att_map.argmax(), att_map.shape)
        print(ova)
        print(att_map[ova[0], ova[1], ova[2]])

        for j in range(6):
            cam_img = np.transpose(imgs[0][j].cpu().numpy(), (1, 2, 0))
            att_map = mean_att[query_location[0]][query_location[1]][j].cpu().numpy()
            plt.title(f'Cam {j}')
            plt.imshow(cam_img)
            # att_map *= 10
            att_map_upsampled = cv2.resize(att_map, dsize=(cam_img.shape[1], cam_img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
            plt.imshow(att_map_upsampled, alpha=att_map_upsampled+.5)
