from dataset import NuscDetDataset, collate_fn
import torch
import os
from functools import partial
from unidepth.models import UniDepthV1
from matplotlib import pyplot as plt
from configs.nuscenes_config import W, H, final_dim, img_conf, ida_aug_conf, bda_aug_conf, CLASSES, names
from loss.depth_loss import silog_loss
from utils import depth_to_disparity, baseline, compute_d1_error, error_to_color
import numpy as np

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


def val_dataloader():
    val_dataset = NuscDetDataset(ida_aug_conf=ida_aug_conf,
                                     bda_aug_conf=bda_aug_conf,
                                     classes=CLASSES,
                                     data_root=data_root,
                                     info_paths=train_info_paths,
                                     is_train=False,
                                     img_conf=img_conf,
                                     num_sweeps=num_sweeps,
                                     sweep_idxes=sweep_idxes,
                                     key_idxes=key_idxes,
                                     return_depth=use_fusion,
                                     use_fusion=use_fusion)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=use_fusion),
            num_workers=4,
            sampler=None,
    )
    return val_loader

def D1_error(predict, gt_depth, focal_length, abs_th=20, rel_th=0.05):
    predict = pred_depth.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    gt_disparity = depth_to_disparity(gt_depth, focal_length=focal_length, baseline=baseline)
    valid_mask = gt_disparity > 0
    predict = predict * valid_mask
    pred_disparity = depth_to_disparity(predict, focal_length=focal_length, baseline=baseline)
    d1_error, error_map = compute_d1_error(gt_disparity, pred_disparity, abs_th, rel_th)
    print(f'D1 Error: {d1_error:.2f}%')

    valid_mask = gt_disparity > 0
    # 计算每个像素的绝对误差 calculate absolute error on pixel wise
    abs_error = np.abs(pred_disparity - gt_disparity)
    # print("abs_err mean", abs_error[valid_mask].mean())
    # 计算相对误差 calculate relative error
    rel_error = np.zeros_like(abs_error)
    # 仅在有效像素上计算相对误差
    rel_error[valid_mask] = abs_error[valid_mask] / gt_disparity[valid_mask]

    error_mask = ((abs_error > abs_th) & (rel_error > rel_th)) & valid_mask
    # 对误差值进行对数变换
    epsilon = 1e-6  # 防止对零取对数
    log_abs_error = np.log(abs_error + epsilon)
    # 将对数误差值归一化到 [0, 1] 范围
    log_abs_error_norm = (log_abs_error - log_abs_error.min()) / (log_abs_error.max() - log_abs_error.min())
    error_color_map = error_to_color(log_abs_error_norm, error_mask, valid_mask)

    return error_color_map, d1_error

def print_list_values_with_names(ret_list, names):
    for name, value in zip(names, ret_list):
        if isinstance(value, torch.Tensor):
            print(f"{name} shape: {value.shape}")
        elif isinstance(value, dict):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value}")

def compute_silog_per_column_torch(prediction: torch.Tensor, target: torch.Tensor, variance_focus: float = 0.85):
    """
    使用 silog_loss 函数计算每一列的SILog误差。

    参数：
        prediction (torch.Tensor): 预测深度图像（H x W）
        target (torch.Tensor): 真实深度图像（H x W）
        variance_focus (float): SILog计算中的方差聚焦参数

    返回：
        np.array: 包含每列SILog值的数组（长度为W）
    """

    prediction = prediction.squeeze()
    target = target.squeeze()
    # 检查输入张量的形状是否一致
    assert prediction.shape == target.shape, "prediction 和 target 的形状必须相同"

    H, W = prediction.shape
    silog_per_column = []

    for j in range(W):
        # 提取第j列
        pred_col = prediction[:, j]
        gt_col = target[:, j]

        # 计算该列的SILog值
        silog_value = silog_loss(pred_col, gt_col, variance_focus=variance_focus)
        silog_per_column.append(silog_value.item())

    return np.array(silog_per_column)

if __name__ == "__main__":
    dataloader = train_dataloader()
    data_iter = iter(dataloader)

    # data index
    data_idx = 60

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

        intrin_mats = single_data[1]['intrin_mats']
        # shape [1, 1, 6, 4, 4]
        focal_length_x = intrin_mats[0, 0, cam_id, 0, 0].item()  # 第一个样本，第一时间步的水平焦距
        focal_length_y = intrin_mats[0, 0, cam_id, 1, 1].item()  # 第一个样本，第一时间步的垂直焦距


        model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)


        predictions = model.infer(img)

        # Metric Depth Estimation
        # pred_depth [1, 1, 256, 704]
        pred_depth = predictions["depth"]
        silog = silog_loss(pred_depth.to(device), depth_gt.to(device).unsqueeze(0), variance_focus=1)
        silog_per_column = compute_silog_per_column_torch(pred_depth.to(device), depth_gt.to(device).unsqueeze(0))
        # d1_error_map, d1_error = D1_error(pred_depth, depth_gt, focal_length_x)


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

        # axs[3].imshow(d1_error_map)
        # axs[3].axis('off')
        # axs[3].set_title('D1 error: %{:.2f}'.format(d1_error))

        fig.suptitle('silog: {:.2f}'.format(silog), fontsize=16, fontweight='bold')
        output_file_depth = os.path.join(output_dir, "img{}_cam{}.png".format(data_idx, cam_id))
        plt.savefig(output_file_depth, bbox_inches='tight')
        # plt.close(fig)
        plt.show()

        plt.figure()
        output_file_column = os.path.join(output_dir, "c_silog_img{}_cam{}.png".format(data_idx, cam_id))
        plt.plot(silog_per_column, label='每列的SILog')
        plt.grid(True)
        plt.title('silog per column')
        plt.xlabel("column")
        plt.ylabel("SILog")
        plt.savefig(output_file_column, bbox_inches='tight')
        # plt.close(fig)
        plt.show()
