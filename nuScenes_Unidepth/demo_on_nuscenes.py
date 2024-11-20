from dataset import NuscDetDataset, collate_fn
import torch
import os
from functools import partial
from unidepth.models import UniDepthV1
from matplotlib import pyplot as plt
from configs.nuscenes_config import W, H, final_dim, img_conf, ida_aug_conf, bda_aug_conf, CLASSES, names
from loss.depth_loss import silog_loss


data_root = '/home/zzhang/work/BEVDepth/data/nuScenes'
train_info_paths = os.path.join(data_root, 'nuscenes_infos_train.pkl')
num_sweeps = 1
sweep_idxes = list()
key_idxes = list()
data_return_depth = True
use_fusion = False

def train_dataloader():
    train_dataset = NuscDetDataset(ida_aug_conf=ida_aug_conf,
                                   bda_aug_conf=bda_aug_conf,
                                   classes=CLASSES,
                                   data_root=data_root,
                                   info_paths=train_info_paths,
                                   is_train=True,
                                   use_cbgs=False,
                                   img_conf=img_conf,
                                   num_sweeps=num_sweeps,
                                   sweep_idxes=sweep_idxes,
                                   key_idxes=key_idxes,
                                   return_depth=data_return_depth,
                                   use_fusion=use_fusion)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        drop_last=True,
        shuffle=False,
        collate_fn=partial(collate_fn,
                           is_return_depth=data_return_depth
                                           or use_fusion),
        sampler=None,
    )
    return train_loader

def print_list_values_with_names(ret_list, names):
    for name, value in zip(names, ret_list):
        if isinstance(value, torch.Tensor):
            print(f"{name} shape: {value.shape}")
        elif isinstance(value, dict):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    dataloader = train_dataloader()
    data_iter = iter(dataloader)

    # data index
    data_idx = 50

    output_dir = './result'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(data_idx):
        data_batch = next(data_iter)

    # unpack batch
    single_data = data_batch

    # Print each data from a batch
    # print_list_values_with_names(single_data, names)

    # camera id in range (0, 5)
    for cam_id in range(6):

        img = single_data[0]
        depth_gt = single_data[6]
        img = img[0][0][cam_id].unsqueeze(0)
        depth_gt = depth_gt[0][0][cam_id].unsqueeze(0)

        model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)


        predictions = model.infer(img)

        # Metric Depth Estimation
        # pred_depth [1, 1, 256, 704]
        pred_depth = predictions["depth"]
        silog = silog_loss(pred_depth.to(device), depth_gt.to(device).unsqueeze(0), variance_focus = 1)


        fig, axs = plt.subplots(3, 1, figsize=(20, 10))
        axs[0].imshow(img.squeeze(0).permute(1, 2, 0).cpu(), cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('img')

        axs[1].imshow(pred_depth.squeeze(0).squeeze(0).cpu(), cmap=plt.get_cmap('inferno_r'))
        axs[1].axis('off')
        axs[1].set_title('predict')

        axs[2].imshow(depth_gt.squeeze(0).squeeze(0).cpu(), cmap=plt.get_cmap('inferno_r'))
        axs[2].axis('off')
        axs[2].set_title('depth gt')

        fig.suptitle('silog: {:.2f}'.format(silog), fontsize=16, fontweight='bold')
        output_file = os.path.join(output_dir, "img{}_cam{}.png".format(data_idx, cam_id))
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        plt.show()
