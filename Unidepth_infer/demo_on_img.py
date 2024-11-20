from unidepth.models import UniDepthV1
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from datasets.kittydata import MyDataset
from configs.option import args
from matplotlib.colors import Normalize
from loss.depth_loss import silog_loss
from utils.functions import focal_length, baseline, depth_to_disparity, compute_d1_error, error_to_color
import glob

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
img_folder = "./img"
# 使用 glob 获取所有 PNG 图片路径
img_paths = glob.glob(f"{img_folder}/*.png")
image_path = img_paths[5]
image_name = os.path.basename(image_path)
save_path = f"./depth/{image_name}"  # 保存路径
os.makedirs(os.path.dirname(save_path), exist_ok=True)

rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W
predictions = model.infer(rgb)
# Metric Depth Estimation
pred_depth = predictions["depth"]

# loss = silog_loss(pred_depth.to(device), gt_depth.unsqueeze(0).to(device))
# print("SILog", loss)


plt.plot(figsize=(20, 10))

plt.imshow(pred_depth.squeeze().to('cpu'), cmap=plt.get_cmap('inferno_r'))
plt.axis('off')
plt.title(image_name)

plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()