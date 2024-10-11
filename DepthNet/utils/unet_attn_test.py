import torch
from models.unet_attention import UNetWithCrossAttention
from configs.option import args
from datasets.kittydata import MyDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    img = torch.randn(1, 3, 352, 704)
    cropped_img = torch.randn(1, 3, 352, 352)

    model = UNetWithCrossAttention(in_channels=3, num_classes=1)
    out = model(img, cropped_img)

    print(out['out'].shape)