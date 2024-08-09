from configs.option import args, parser
from datasets.kittydata import MyDataset
import matplotlib.pyplot as plt
import numpy as np
from generate_dense_depth import interpolate_depth_map

if __name__ == "__main__":
    args.dataset = "KITTI"
    args.trainfile_kitti = "../datasets/eigen_train_files_with_gt_dense.txt"
    train_set = MyDataset(args, train=True)
    img, gt_depth, _ = train_set[0]
    dense_depth = interpolate_depth_map(gt_depth)
    print(f'img shape: {img.shape},  gt_depth shape: {gt_depth.shape}, dense depth shape: {dense_depth.shape}')
    print(f" length of training set: {len(train_set)}")
    print(dense_depth)
    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(np.transpose(gt_depth, (1, 2, 0)))
    axs[0].axis('off')
    axs[0].set_title('gt_depth')

    axs[1].imshow(np.transpose(dense_depth, (1, 2, 0)))
    axs[1].axis('off')
    axs[1].set_title('dense map')

    axs[2].imshow(np.transpose(img, (1, 2, 0)))
    axs[2].axis('off')
    axs[2].set_title('Augumented Img')
    plt.show()