import torch

def gaussian_noise(gt, mean=0., std=1.):
    device = gt.device
    noise = torch.normal(mean, std, size=gt.shape).to(device)
    gt_mask_noisy = gt.clone()
    gt_mask_noisy = gt_mask_noisy.float()
    gt_mask_noisy[gt > 0] += noise[gt > 0]

    # Clamp to avoid negative distances
    gt_mask_noisy = torch.clamp(gt_mask_noisy, min=0)
    return gt_mask_noisy