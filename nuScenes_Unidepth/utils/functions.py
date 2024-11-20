import numpy as np

focal_length = 721.5377  # 焦距，单位：像素（来自KITTI）
baseline = 0.532722  # 基线长度，单位：米（来自KITTI）
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