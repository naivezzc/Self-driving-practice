import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.functional as TF
from torch import Tensor


def silog_loss(prediction: Tensor, target: Tensor, variance_focus: float = 0.85) -> float:
    """
    Scale invariant logarithmic error [log(m)*100] (for more info click on the formula below)
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (prediction > 0)

    # SILog
    d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    n = target[non_zero_mask].shape[0]

    if n == 0:
        return torch.tensor(0.0).to(d.device)

    # print(f"predict d range: min={prediction.min()}, max={prediction.max()}")
    # print(f"gt d range: min={target.min()}, max={target.max()}")

    # return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0
    return torch.sqrt((d ** 2).mean() - variance_focus * 1 / (n**2) * (d.sum() ** 2)) * 100
