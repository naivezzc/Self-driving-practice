from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from configs.option import args
from datasets.kittydata import MyDataset
from models.unet import UNet
from loss.depth_loss import DepthLoss, Masked_depthLoss, silog_loss
def test(model, dataloader, device, mask_flag=False):
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)
    loss_fn = silog_loss

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing Progress")
        for i, data in enumerate(progress_bar):
            inputs, targets, _ = data
            inputs, targets = inputs.to(device), targets.to(device)

            # Filter data missing Ground Truth
            if targets.shape == torch.Size([1]):
                continue

            mask = targets > 0.001

            outputs = model(inputs)['out']

            loss = silog_loss(targets, outputs, variance_focus=0.1)

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

    return running_loss / num_batches


if __name__ == "__main__":
    test_set = MyDataset(args, train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = args.weights
    mask_flag = False

    model = UNet(in_channels=3, num_classes=1).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    test_loss = test(model, test_loader, device, mask_flag)

    print(f"SILog: {test_loss:.3f}")


