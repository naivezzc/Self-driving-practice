from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from configs.option import args
from datasets.kittydata import MyDataset
from models.unet import UNet
from models.unet_attention import UNetWithCrossAttention
from loss.depth_loss import DepthLoss, Masked_depthLoss, silog_loss
from utils.noise import gaussian_noise


def test(model, dataloader, device, mask_flag=False, attn = False, add_noise = False, noise_mean=0.0, noise_std=0.0):
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)
    loss_fn = silog_loss

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing Progress")
        for i, data in enumerate(progress_bar):
            inputs, targets, _, crop_img, crop_gt = data
            inputs, targets, crop_img, crop_gt = inputs.to(device), targets.to(device), crop_img.to(device), crop_gt.to(device)
            if add_noise:
                crop_gt = gaussian_noise(crop_gt, mean=noise_mean, std=noise_std)

            # Filter data missing Ground Truth
            if targets.shape == torch.Size([1]):
                continue

            if crop_img.shape == torch.Size([1]):
                # print(targets)
                continue

            if crop_gt.shape == torch.Size([1]):
                continue

            mask = targets > 0.001

            if attn == False:
                outputs = model(inputs)['out']
            else:
                # print(crop_gt, crop_img)
                # print(crop_gt.shape)
                outputs = model(inputs, crop_img, crop_gt)['out']

            loss = silog_loss(targets, outputs, variance_focus=1)

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

    return running_loss / num_batches


if __name__ == "__main__":
    test_set = MyDataset(args, train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # weight_path = args.weights
    weight_path = 'weights/unet_25per_512_std25.pth'
    attn = True
    add_noise = True
    noise_mean = 0.0
    noise_std = 10.0
    # model = UNet(in_channels=3, num_classes=1).to(device)
    img_size = (352, 704)
    crop_size = (352, 176)
    qk_dim = 512
    model = UNetWithCrossAttention(in_channels=3, num_classes=1,img_size=img_size, crop_size=crop_size, attn_dim_qk=qk_dim, attn_dim_v=512).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    test_loss = test(model, test_loader, device, attn=attn, add_noise=add_noise, noise_mean=noise_mean, noise_std=noise_std)

    print(f"SILog: {test_loss:.3f}")


