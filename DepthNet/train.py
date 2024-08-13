import torch
from models.unet import UNet
from models.mobile_unet import MobileV3Unet
from loss.depth_loss import DepthLoss, Masked_depthLoss
from configs.option import args
from datasets.kittydata import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datetime
import json

def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs, mask_flag = False):
    model.train()
    running_loss = 0.0
    sum_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} / {num_epochs}")
    for i, data in enumerate(progress_bar):
        inputs, targets, _ = data
        inputs, targets = inputs.to(device), targets.to(device)
        mask = targets > 0.001

        optimizer.zero_grad()
        outputs = model(inputs)['out']
        if mask_flag == False:
            loss = criterion(targets, outputs)
        else:
            loss = criterion(targets, outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        sum_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (i + 1)})
        if i % 10 == 9:  # Print every 10 mini-batches
            # print(f"[{i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0
    return sum_loss / len(dataloader)



if __name__ == "__main__":
    args.dataset = "KITTI"
    train_set = MyDataset(args, train=True)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_flag = True
    num_epochs = args.epochs
    lr = args.lr

    # model = UNet(in_channels=3, num_classes=1).to(device)
    model = MobileV3Unet(num_classes=1).to(device)
    if mask_flag == False:
        criterion = DepthLoss()
    else:
        criterion = Masked_depthLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_his = []
    for epoch in range(num_epochs):
        epoch_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, mask_flag)
        loss_his.append(epoch_loss)

    # Save the trained model
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M")
    save_path = f"./weights/unet_{formatted_time}.pth"
    loss_path = f"./log/unet_{formatted_time}.json"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

    # Save loss curve
    with open(loss_path, 'a') as f:
        json.dump(loss_his, f)