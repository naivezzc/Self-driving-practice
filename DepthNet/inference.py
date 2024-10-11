from configs.option import args
from datasets.kittydata import MyDataset
from models.unet import UNet
from models.mobile_unet import MobileV3Unet
from models.unet_attention import UNetWithCrossAttention
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from loss.depth_loss import silog_loss
from matplotlib.colors import Normalize
from datasets.transform_list import CenterCropNumpy
import os


def depth_to_disparity(depth, focal_length, baseline):
    # 避免除以零
    valid_mask = depth > 0
    disparity = np.zeros_like(depth)
    disparity[valid_mask] = (focal_length * baseline) / depth[valid_mask]
    return disparity

def compute_d1_error(gt_disp, pred_disp):
    # 有效像素掩码（真实视差大于0）
    mask = gt_disp > 0

    # 计算绝对误差
    abs_diff = np.abs(gt_disp - pred_disp)

    # 误差条件
    error_mask = (abs_diff > 3) & (abs_diff > 0.05 * gt_disp)

    # 计算D1误差百分比
    error_pixels = np.sum(error_mask & mask)
    total_pixels = np.sum(mask)
    d1_error = (error_pixels / total_pixels) * 100

    # 生成误差图
    error_map = np.zeros_like(gt_disp)
    error_map[error_mask & mask] = 1  # 标记错误像素

    return d1_error, error_map

def error_to_color(error_norm, error_mask, valid_mask):
    # 初始化颜色图像
    H, W = error_norm.shape
    color_image = np.zeros((H, W, 3), dtype=np.float32)

    # 正确预测的像素（蓝色）
    color_image[..., 2] = (~error_mask) & valid_mask  # 蓝色通道

    # 预测错误的像素（红色调，根据误差大小）
    color_image[..., 0] = error_norm * error_mask     # 红色通道

    # 遮挡和无效像素（黑色）
    color_image[~valid_mask] = 0

    return color_image

if __name__ == "__main__":
    weight_path = "./weights/unet_2024_09_27_04_48.pth"
    use_attn = True
    # weight_path = args.weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    focal_length = 721.5377  # 焦距，单位：像素（来自KITTI）
    baseline = 0.532722      # 基线长度，单位：米（来自KITTI）


    test_set = MyDataset(args, train=False, return_filename=True)
    aug_img, gt_depth, _, filename, crop_rgb, crop_gt = test_set[6]
    aug_img, gt_depth, crop_rgb, crop_gt = aug_img.to(device), gt_depth.to(device), crop_rgb.to(device), crop_gt.to(device)
    crop_rgb, crop_gt = crop_rgb.unsqueeze(0), crop_gt.unsqueeze(0)
    aug_img = aug_img.unsqueeze(0)
    img = Image.open(filename)
    width, height = img.size  # width=704, height=352

    # 计算裁剪区域的坐标
    new_width = 704
    left = (width - new_width) / 2
    top = 0
    right = left + new_width
    bottom = height

    # 将坐标转换为整数
    box = (int(left), int(top), int(right), int(bottom))

    # 裁剪图片
    img = img.crop(box)

    # print(filename)

    # model = UNet(in_channels=3, num_classes=1).to(device)
    # model = MobileV3Unet(num_classes=1).to(device)
    # model = UNetWithCrossAttention(in_channels=3, num_classes=1).to(device)
    img_size = (352, 704)
    crop_size = (352, 176)
    if use_attn:
        model = UNetWithCrossAttention(in_channels=3, num_classes=1, img_size=img_size, crop_size=crop_size).to(device)
    else:
        model = UNet(in_channels=3, num_classes=1).to(device)

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("shape", aug_img.shape, crop_rgb.shape, crop_gt.shape, gt_depth.shape)

    with torch.no_grad():
        if use_attn:
            output = model(aug_img, crop_rgb, crop_gt)['out']
        else:
            output = model(aug_img)['out']
        loss = silog_loss(output, gt_depth.unsqueeze(0))
        print("SILog", loss)

    print(f"Output shape: {output.shape}")

    predict = output.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    aug_img = aug_img.squeeze().cpu().numpy()

    gt_disparity = depth_to_disparity(gt_depth, focal_length=focal_length, baseline=baseline)
    valid_mask = gt_disparity > 0
    predict = predict * valid_mask
    pred_disparity = depth_to_disparity(predict, focal_length=focal_length, baseline=baseline)
    d1_error, error_map = compute_d1_error(gt_disparity, pred_disparity)
    print("pre_disparity", pred_disparity)
    print("gt_disparity", gt_disparity)
    print(f'D1 Error: {d1_error:.2f}%')


    valid_mask = gt_disparity > 0
    # 计算每个像素的绝对误差 calculate absolute error on pixel wise
    abs_error = np.abs(pred_disparity - gt_disparity)
    # 计算相对误差 calculate relative error
    rel_error = np.zeros_like(abs_error)
    # 仅在有效像素上计算相对误差
    rel_error[valid_mask] = abs_error[valid_mask] / gt_disparity[valid_mask]

    error_mask = ((abs_error > 3) & (rel_error > 0.05)) & valid_mask
    # 对误差值进行对数变换
    epsilon = 1e-6  # 防止对零取对数
    log_abs_error = np.log(abs_error + epsilon)
    # 将对数误差值归一化到 [0, 1] 范围
    log_abs_error_norm = (log_abs_error - log_abs_error.min()) / (log_abs_error.max() - log_abs_error.min())

    error_color_map = error_to_color(log_abs_error_norm, error_mask, valid_mask)

    error_visual = np.zeros((gt_disparity.shape[0], gt_disparity.shape[1], 3))
    error_visual[..., 2] = (error_map == 0)  # 正确的像素显示为蓝色
    error_visual[..., 0] = error_map  # 错误的像素显示为红色

    print(f"predict shape: {predict.shape}")

    # 计算全局最小值和最大值
    global_min = min(np.min(predict), np.min(gt_depth))
    global_max = max(np.max(predict), np.max(gt_depth))
    norm = Normalize(vmin=global_min, vmax=global_max)


    fig, axs = plt.subplots(6, 1, figsize=(20, 10))
    axs[0].imshow(gt_depth,cmap=plt.get_cmap('inferno_r'), norm=norm)
    axs[0].axis('off')
    axs[0].set_title('gt_depth')

    axs[1].imshow(predict, cmap=plt.get_cmap('inferno_r'), norm=norm)
    axs[1].axis('off')
    axs[1].set_title('predict')

    # axs[2].imshow(np.transpose(aug_img, (1, 2, 0)))
    # axs[2].axis('off')
    # axs[2].set_title('Augmented img')

    axs[2].imshow(img)
    axs[2].axis('off')
    axs[2].set_title('img')

    axs[3].imshow(gt_disparity, cmap='plasma')
    axs[3].axis('off')
    axs[3].set_title('gt_disparity')

    axs[4].imshow(pred_disparity, cmap='plasma')
    axs[4].axis('off')
    axs[4].set_title('pred_disparity')

    axs[5].imshow(error_color_map)
    axs[5].axis('off')
    axs[5].set_title('D1 error')

    plt.show()
