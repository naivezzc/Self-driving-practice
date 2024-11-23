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

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
# img_dir = '/home/zzhang/ssd2/dataset/KITTI/2011_10_03/2011_10_03_drive_0058_sync/image_03/data'
# img_name = '0000000000.png'
# img_path = os.path.join(img_dir, img_name)
# rgb = torch.from_numpy(np.array(Image.open(img_path))).permute(2, 0, 1) # C, H, W

test_set = MyDataset(args, train=False, return_filename=True)
img_id = 122
aug_img, gt_depth, _, filename, crop_rgb, crop_gt = test_set[img_id]

# Center crop image to (352, 704)
img = Image.open(filename)
width, height = img.size  # width=704, height=352
new_width = 704
left = (width - new_width) / 2
top = 0
right = left + new_width
bottom = height
box = (int(left), int(top), int(right), int(bottom))
img = img.crop(box)

predictions = model.infer(aug_img)

# Metric Depth Estimation
pred_depth = predictions["depth"]

loss = silog_loss(pred_depth.to(device), gt_depth.unsqueeze(0).to(device))
print("SILog", loss)

predict = pred_depth.squeeze().cpu().numpy()
gt_depth = gt_depth.squeeze().cpu().numpy()
aug_img = aug_img.squeeze().cpu().numpy()
gt_crop = crop_gt.squeeze().cpu().numpy()

gt_disparity = depth_to_disparity(gt_depth, focal_length=focal_length, baseline=baseline)
valid_mask = gt_disparity > 0
predict = predict * valid_mask
pred_disparity = depth_to_disparity(predict, focal_length=focal_length, baseline=baseline)
d1_error, error_map = compute_d1_error(gt_disparity, pred_disparity)
print(f'D1 Error: {d1_error:.2f}%')

valid_mask = gt_disparity > 0
# calculate absolute error on pixel wise
abs_error = np.abs(pred_disparity - gt_disparity)
# calculate relative error
rel_error = np.zeros_like(abs_error)
# Compute relative error only on valid pixels
rel_error[valid_mask] = abs_error[valid_mask] / gt_disparity[valid_mask]

error_mask = ((abs_error > 3) & (rel_error > 0.05)) & valid_mask
# Apply logarithmic transformation to error values
epsilon = 1e-6  # Prevent logarithm of zero
log_abs_error = np.log(abs_error + epsilon)
# Normalize logarithmic error values to the range [0, 1]
log_abs_error_norm = (log_abs_error - log_abs_error.min()) / (log_abs_error.max() - log_abs_error.min())
error_color_map = error_to_color(log_abs_error_norm, error_mask, valid_mask)

print(f"predict shape: {predict.shape}")

# Point Cloud in Camera Coordinate
xyz = predictions["points"]
# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

global_min = min(torch.min(pred_depth), np.min(gt_depth))
global_max = max(torch.max(pred_depth), np.max(gt_depth))

norm = Normalize(vmin=global_min, vmax=global_max)
# norm = Normalize(vmin=-43.016045, vmax= 81.48552)

print(global_min, global_max)

fig, axs = plt.subplots(4, 1, figsize=(20, 10))
axs[0].imshow(gt_depth.squeeze(), cmap=plt.get_cmap('inferno_r'), norm=norm)
axs[0].axis('off')
axs[0].set_title('gt_depth')

axs[1].imshow(pred_depth.squeeze().to('cpu'), cmap=plt.get_cmap('inferno_r'), norm=norm)
axs[1].axis('off')
axs[1].set_title('predict')

axs[2].imshow(error_color_map)
axs[2].axis('off')
axs[2].set_title('D1 error')

axs[3].imshow(img)
axs[3].axis('off')
axs[3].set_title('img')

save_dir = './result'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{img_id}_d1error_{d1_error:.2f}_silog_{loss:.2f}.png')

plt.savefig(save_path)
plt.show()



