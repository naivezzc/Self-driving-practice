import torch
from torch.utils.data import Dataset

class Mydataset(Dataset):
    def __init__(self):
        self.data = torch.arange(0, 20)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)