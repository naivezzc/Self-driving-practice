from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from configs.option import args
from loss.depth_loss import silog_loss
from unidepth.models import UniDepthV1
import os
from dataset import NuscDetDataset, collate_fn
import torch
from functools import partial
from configs.nuscenes_config import W, H, final_dim, img_conf, ida_aug_conf, bda_aug_conf, CLASSES, names

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

def eval_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)
    loss_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing Progress")
        for i, data in enumerate(progress_bar):
            (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = data
            sweep_imgs, depth_labels = sweep_imgs.to(device), depth_labels.to(device)

            mask = depth_labels > 0.001

            # sweep_imgs [1, 1, 6, 3, 256, 704]
            # depth_labels [1, 1, 6, 256, 704]
            # each sample have 6 images from different cameras respectively
            for img_id in range(6):
                predictions = model.infer(sweep_imgs[0,:,img_id,:, :, :])
                outputs = predictions["depth"]
                loss = silog_loss(depth_labels[:, :, img_id, :, :], outputs, variance_focus=1)
                running_loss += loss.item()
                loss_list.append(loss.item())
                progress_bar.set_postfix({'loss': running_loss / ((i + 1) * 6)})

    return running_loss / (num_batches * 6), loss_list

def box_plot(loss_list):
    '''
    :param loss_list:  1 dimension list (e.g. [5.1, 5.2, 5.3, 5.4, 5.5, 5.6])
    '''
    plt.figure(figsize=(6, 8))
    plt.boxplot(loss_list, vert=True, patch_artist=True)

    plt.title("Boxplot of SILog on nuScenes dataset", fontsize=14)
    plt.ylabel("SILog Values", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

if __name__ == "__main__":
    test_loader = train_dataloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to(device)

    test_loss, loss_list = eval_model(model, test_loader, device)
    box_plot(loss_list)

    print(f"SILog: {test_loss:.3f}")