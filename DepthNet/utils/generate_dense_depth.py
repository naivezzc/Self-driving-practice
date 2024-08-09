import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def interpolate_depth_map(gt_depth):
    # gt_depth 是一个形状为 (1, 352, 704) 的 torch.Tensor
    gt_depth = gt_depth.squeeze(0)  # 移除batch维度, 变成 (352, 704)

    # 将深度图转换为numpy数组以便处理
    gt_depth_np = gt_depth.numpy()

    # 找到深度值为零的索引
    mask = gt_depth_np == 0

    # 创建一个网格
    grid_y, grid_x = np.meshgrid(np.arange(gt_depth_np.shape[0]), np.arange(gt_depth_np.shape[1]), indexing='ij')

    # 获取非零深度值的索引和值
    valid_x = grid_x[~mask]
    valid_y = grid_y[~mask]
    valid_depth = gt_depth_np[~mask]

    # 进行线性插值
    interpolated_depth = griddata((valid_y, valid_x), valid_depth, (grid_y, grid_x), method='linear')

    # 如果插值结果中有NaN值，用最近邻插值法填充这些NaN值
    nan_mask = np.isnan(interpolated_depth)
    interpolated_depth[nan_mask] = griddata((valid_y, valid_x), valid_depth, (grid_y[nan_mask], grid_x[nan_mask]), method='nearest')

    # 将插值后的深度图转换为 torch.Tensor
    interpolated_depth = torch.from_numpy(interpolated_depth).unsqueeze(0)

    return interpolated_depth