import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))