import torch
from torch.utils.data import DataLoader
from dataset import Mydataset
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = Mydataset()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=4)

    # Length of dataset
    print(len(dataset))
    print(len(dataloader))

    print(dataset[4])

    for batch in dataloader:
        print(batch)
