import torch
from models.unet import UNet
from loss.depth_loss import DepthLoss
from loss.depth_loss import relative_mse_loss

if __name__ == "__main__":
    unet = UNet(in_channels=3, num_classes=1)
    loss_fn = DepthLoss()

    input = torch.rand(1, 3, 256, 256)
    gt = torch.rand(1, 1, 256, 256)

    pred = unet(input)['out']

    print(f"loss: {loss_fn(gt, pred)}")
