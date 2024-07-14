from dataset import NuscDetDataset, collate_fn
import torch
import os
from functools import partial

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

data_root = '/home/zhang_hm/work/BEVDepth/data/nuScenes'
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


names = [
    "imgs_batch",
    "mats_dict",
    "timestamps_batch",
    "img_metas_batch",
    "gt_boxes_batch",
    "gt_labels_batch",
    'depth'
]

if __name__ == "__main__":
    dataloader = train_dataloader()
    data_iter = iter(dataloader)

    # 使用 next() 获取一个 batch 的数据
    data_batch = next(data_iter)

    # 获取单个数据点
    single_data = data_batch

    print_list_values_with_names(single_data, names)
    print(len(single_data))
