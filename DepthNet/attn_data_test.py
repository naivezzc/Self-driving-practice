import torch
from models.unet_attention import UNetWithCrossAttention
from configs.option import args
from datasets.kittydata import MyDataset
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16


if __name__ == '__main__':
    # img = torch.randn(1, 3, 352, 704)
    # cropped_img = torch.randn(1, 3, 352, 352)

    train_set = MyDataset(args, train=True)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

    for data in train_loader:
        inputs, targets, _, crop_img, crop_gt = data
        inputs, targets, crop_img, crop_gt = inputs.cuda(), targets.cuda(), crop_img.cuda(), crop_gt.cuda()
        break

    print("shape", inputs.shape, crop_img.shape, crop_gt.shape)
    img_size = (inputs.shape[2], inputs.shape[3])
    crop_size = (crop_img.shape[2], crop_img.shape[3])
    print(img_size, crop_size)

    model = UNetWithCrossAttention(in_channels=3, num_classes=1, img_size=img_size, crop_size=crop_size).cuda()
    out = model(inputs, crop_img, crop_gt)

    print('output depth shape', out['out'].shape)
    print('gt_crop', crop_gt.shape)