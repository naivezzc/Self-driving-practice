import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.functional as TF
from torch import Tensor
import pytorch_ssim


class DepthLoss(nn.Module):
    def __init__(self, w1=1.0, w2=0.00, w3=0.1):
        super(DepthLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, y_true, y_pred):
        # Depth loss
        l_depth = torch.mean(torch.abs(y_pred - y_true), dim=-1)

        # Edge loss for sharp edges
        dy_true, dx_true = self.image_gradients(y_true)
        dy_pred, dx_pred = self.image_gradients(y_pred)
        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=-1)

        # Structural similarity loss (SSIM)
        # Details at https://github.com/Po-Hsun-Su/pytorch-ssim
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        l_ssim = torch.clamp((1 - ssim_loss(y_pred, y_true)) * 0.5, 0, 1)

        # Weighted sum of loss
        loss = (self.w1 * l_ssim.mean()) + (self.w2 * l_edges.mean()) + (self.w3 * l_depth.mean())
        # print(f"loss 1: {self.w1 * l_ssim.mean()}, loss 2: {self.w2 * l_edges.mean()}, loss 3: {self.w3 * l_depth.mean()}")
        return loss

    def image_gradients(self, image):
        # gradients on x axis
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        # gradients on y axis
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

        # keep dim same as original images
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

        return grad_x, grad_y


class Masked_depthLoss(nn.Module):
    """
    Compute Depth loss only for masked pixels (omit pixels without lidar depth).

    Args:
        w1 (Tensor): Weight of the Structural similarity loss (SSIM). Defaults to 1.0.
        w2 (Tensor): Weight of gradient loss.(gradients loss only can be used with dense depth map), defaults to 0.
        w3 (bool): Weight of Mean Absolute Error loss. Defaults to 0.1.

    Returns:
        float: loss.
    """
    def __init__(self, w1=1.0, w2=0.00, w3=0.1):
        super(Masked_depthLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, y_true, y_pred, mask):
        # Depth loss
        masked_y_pred = torch.zeros_like(y_pred)
        masked_y_pred[mask] = y_pred[mask]
        masked_y_true = torch.zeros_like(y_true)
        masked_y_true[mask] = y_true[mask]

        # print(masked_y_pred)
        # print(masked_y_true)
        # print(y_true)

        l_depth = torch.mean(torch.abs(masked_y_pred - masked_y_true), dim=-1)

        # Edge loss for sharp edges
        dy_true, dx_true = self.image_gradients(masked_y_true)
        dy_pred, dx_pred = self.image_gradients(masked_y_pred)
        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=-1)

        # Structural similarity loss (SSIM)
        # Details at https://github.com/Po-Hsun-Su/pytorch-ssim
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        l_ssim = torch.clamp((1 - ssim_loss(masked_y_pred, masked_y_true)) * 0.5, 0, 1)

        # Weighted sum of loss
        loss = (self.w1 * l_ssim.mean()) + (self.w2 * l_edges.mean()) + (self.w3 * l_depth.mean())
        # print(f"loss 1: {self.w1 * l_ssim.mean()}, loss 2: {self.w2 * l_edges.mean()}, loss 3: {self.w3 * l_depth.mean()}")
        return loss

    def image_gradients(self, image):
        # gradients on x axis
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        # gradients on y axis
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

        # keep dim same as original images
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

        return grad_x, grad_y




def relative_mse_loss(prediction: Tensor, target: Tensor, mask_zero: bool = False) -> float:
    """
    Compute MSE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MSE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mse = F.mse_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    target_mse = F.mse_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))

    return pred_mse / target_mse * 100


def relative_mae_loss(prediction: Tensor, target: Tensor, mask_zero: bool = True):
    """
    Compute MAE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MAE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mae = F.l1_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    target_mae = F.l1_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))

    return pred_mae / target_mae * 100


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

    # return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0
    return torch.sqrt((d ** 2).mean() - variance_focus * 1 / (n**2) * (d.sum() ** 2)) * 10.0
