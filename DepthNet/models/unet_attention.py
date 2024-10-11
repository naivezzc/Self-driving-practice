import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Attn_Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Attn_Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x_main, x_crop, crop_depth):
        B, N_main, C = x_main.shape
        N_crop = x_crop.shape[1]
        N_depth = crop_depth.shape[1]

        qkv_main = self.qkv(x_main).reshape(B, N_main, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_main, k_main, v_main = qkv_main[0], qkv_main[1], qkv_main[2]

        qkv_crop = self.qkv(x_crop).reshape(B, N_crop, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_crop, k_crop, v_crop = qkv_crop[0], qkv_crop[1], qkv_crop[2]

        qkv_depth = self.qkv(crop_depth).reshape(B, N_depth, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_depth, k_depth, v_depth = qkv_depth[0], qkv_depth[1], qkv_depth[2]

        # print("QKV size", q_main.shape, k_crop.shape, v_depth.shape)
        # Cross-attention: query from main image, keys and values from crop
        attn = (q_main @ k_crop.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_depth).transpose(1, 2).reshape(B, N_main, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Patch_Embed(nn.Module):
    def __init__(self, img_size=(22, 44), patch_size=2, in_c=512, embed_dim=512, norm_layer=None):
        super().__init__()
        # img_size = img_size
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] / patch_size[0], img_size[1] / patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, W, H = x.shape
        assert W == self.img_size[0] and H == self.img_size[1], \
            "img size{}{} does not match input size{}{}".format(W, H, self.img_size[0], self.img_size[1])
        # flatten B,C,W,H -> B,C,WH
        # transpose B,C,WH -> B,WH,C
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class UNetWithCrossAttention(nn.Module):
    def __init__(self, img_size, crop_size, in_channels, num_classes, attn_dim=512, num_heads=8, bilinear: bool = True, base_c: int = 64, patch_size=1, drop_ratio=0.):
        super(UNetWithCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.patch_size = patch_size
        # Image down sampled by 16X
        # If the img is (W, H) the feature will shape (W/16, H/16)
        self.feature_size = (img_size[0] // 16, img_size[1] // 16)
        self.crop_feature_size = (crop_size[0] // 16, crop_size[1] // 16)
        self.num_q = (img_size[0] // 16) * (img_size[1] // 16)
        self.num_kv = (crop_size[0] // 16) * (crop_size[1] // 16)

        # Share Encoder
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # Cross-Attention机制
        self.cross_attention = Attention(dim=attn_dim, num_heads=num_heads, qkv_bias=False)
        self.pos_embed_q = nn.Parameter(torch.zeros(1, self.num_q, attn_dim))  # Learnable positional embedding for Q
        self.pos_embed_kv = nn.Parameter(torch.zeros(1, self.num_kv, attn_dim))  # Learnable positional embedding for K and V
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # Decoder
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        # Patch Embedding
        self.patch_feature = Patch_Embed(img_size=self.feature_size, patch_size=patch_size, in_c=512, embed_dim=attn_dim)
        self.patch_crop_feature = Patch_Embed(img_size=self.crop_feature_size, patch_size=patch_size, in_c=512, embed_dim=attn_dim)
        self.patch_depth = Patch_Embed(crop_size, patch_size=patch_size*16, in_c=1, embed_dim=attn_dim)

        # Attention feature up sampling
        self.attn_up1 = Attn_Up(base_c * 8, base_c * 4, bilinear)
        self.attn_up2 = Attn_Up(base_c * 4, base_c * 2, bilinear)
        self.attn_up3 = Attn_Up(base_c * 2, base_c, bilinear)
        self.attn_up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x_main: torch.Tensor, x_crop: torch.Tensor, crop_depth:torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoding image and cropped image
        x1_main = self.in_conv(x_main)
        x2_main = self.down1(x1_main)
        x3_main = self.down2(x2_main)
        x4_main = self.down3(x3_main)
        x5_main = self.down4(x4_main)

        x1_crop = self.in_conv(x_crop)
        x2_crop = self.down1(x1_crop)
        x3_crop = self.down2(x2_crop)
        x4_crop = self.down3(x3_crop)
        x5_crop = self.down4(x4_crop)

        # print("x5", x5_main.shape, x5_crop.shape)
        # in_c = x5_main.shape[1]
        # embed_dim = x5_main.shape[1]
        # x5_main_size = (x5_main.shape[2], x5_main.shape[3])
        # x5_crop_size = (x5_crop.shape[2], x5_crop.shape[3])
        # depth_size = (crop_depth.shape[2], crop_depth.shape[3])


        # Patch_feature =  Patch_Embed(img_size=x5_main_size, patch_size= self.patch_size, in_c=in_c, embed_dim=embed_dim).to('cuda:0')
        # Patch_feature_crop = Patch_Embed(img_size=x5_crop_size, patch_size= self.patch_size, in_c=in_c, embed_dim=embed_dim).to('cuda:0')
        # Patch_depth = Patch_Embed(img_size=depth_size, patch_size=self.patch_size*16, in_c= self.patch_size, embed_dim=embed_dim).to('cuda:0')
        #
        # x5_main_token = Patch_feature(x5_main)
        # x5_crop_token = Patch_feature_crop(x5_crop)
        # depth_token = Patch_depth(crop_depth)

        x5_main_token = self.patch_feature(x5_main)
        x5_crop_token = self.patch_crop_feature(x5_crop)
        depth_token = self.patch_depth(crop_depth)

        q = self.pos_drop(x5_main_token + self.pos_embed_q)
        k = self.pos_drop(x5_crop_token + self.pos_embed_kv)
        v = self.pos_drop(depth_token + self.pos_embed_kv)

        # print("x5 token", x5_main_token.shape, x5_crop_token.shape, depth_token.shape)
        # print("add pos embed token", q.shape, k.shape, v.shape)

        # Cross-attention between main image and crop
        x_attended = self.cross_attention(q, k, v)

        # print("attention out shape", x_attended.shape)
        B, C, H, W = x5_main.shape
        x_attended = x_attended.transpose(1, 2).view(B, 512, H, W)
        # print("attention out shape(reshaped)", x_attended.shape)
        # print("x1_main shape", x1_main.shape)

        # Up sampling attention features
        # x_attended = self.attn_up1(x_attended)
        # x_attended = self.attn_up2(x_attended)
        # x_attended = self.attn_up3(x_attended)
        # x_attended = self.attn_up4(x_attended)

        #Add attention feature
        x5_main = x_attended + x5_main


        # 解码
        x = self.up1(x5_main, x4_main)
        x = self.up2(x, x3_main)
        x = self.up3(x, x2_main)
        x = self.up4(x, x1_main)
        logits = self.out_conv(x)

        return {"out": logits}
