from configs.option import args
from datasets.kittydata import MyDataset
from models.unet import UNet
from models.mobile_unet import MobileV3Unet
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    # weight_path = "./weights/unet_add_mask.pth"
    weight_path = args.weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_set = MyDataset(args, train=False, return_filename=True)
    aug_img, gt_depth, _, filename = test_set[4]
    aug_img, gt_depth = aug_img.to(device), gt_depth.to(device)
    aug_img = aug_img.unsqueeze(0)
    img = Image.open(filename)

    # print(filename)

    # model = UNet(in_channels=3, num_classes=1).to(device)
    model = MobileV3Unet(num_classes=1).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(aug_img)['out']

    print(f"Output shape: {output.shape}")

    predict = output.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    aug_img = aug_img.squeeze().cpu().numpy()

    print(f"predict shape: {predict.shape}")

    fig, axs = plt.subplots(3, 1, figsize=(20, 10))
    axs[0].imshow(gt_depth,cmap=plt.get_cmap('inferno_r'))
    axs[0].axis('off')
    axs[0].set_title('gt_depth')

    axs[1].imshow(predict, cmap=plt.get_cmap('inferno_r'))
    axs[1].axis('off')
    axs[1].set_title('predict')

    # axs[2].imshow(np.transpose(aug_img, (1, 2, 0)))
    # axs[2].axis('off')
    # axs[2].set_title('Augmented img')

    axs[2].imshow(img)
    axs[2].axis('off')
    axs[2].set_title('img')

    plt.show()
