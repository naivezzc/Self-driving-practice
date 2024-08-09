# Monocular depth estimation on KITTI Dataset

## Introduction

## Setup
This software depends on the following Python packages:
```
tqdm
json
matplotlib
torch==2.3.0
torchvision==0.18.0
```

All can be installed using pip install. The PyTorch version used for development was 2.3; it can be installed following instructions [here](https://pytorch.org/get-started/previous-versions/).

CUDA is recommended for best performances. Version 11.7 was used during development

## Data Preparation
- Donwload raw data set from [KITTI website](https://www.cvlibs.net/datasets/kitti/raw_data.php) (note: you can use  raw dataset download script (1 MB) )

- Download annotations(annotated depth maps data set (14 GB)) from [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
- Move all files from /data_depth_annotated/val to /data_depth_annotated/train
- Delete the /data_depth_annotated/val directory, then duplicate /data_depth_annotated/train and rename it as /data_depth_annotated/val


## Usage
### Quick test
To quickly try out the code:
```bash
  python3 inference.py --weights [your weight path]
  ```

### Training

Training is launched with the following command
```bash
  python3 train.py --epochs 20 --batch_size 8 --lr 0.0001
  ```

### Evaluating 
```bash
  python3 eval.py --weights [your weight path]
  ```