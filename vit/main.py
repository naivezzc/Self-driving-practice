import torch
from model import Patch_Embed, Attention, MLP, Block

if __name__ == '__main__':
    inputs = torch.rand([1, 3, 224, 224])
    Preprocessor = Patch_Embed(224, 16, 3, 768, norm_layer=None)
    embed = Preprocessor(inputs)
    print("embed shape: {}".format(embed.shape))

    attn = Attention(dim=768, num_heads=8, qkv_scale=None, qkv_bias=False, attention_drop_ratio=0, proj_drop_ratio=0)
    attn_feature = attn(embed)
    print("attn feature shape: {}".format(attn_feature.shape))

    mlp = MLP(in_features=768, hidden_features=768*4, out_features=768, drop=0.)
    mlp_feature = mlp(attn_feature)
    print("mlp feature shape: {}".format(mlp_feature.shape))

    blocks = Block(dim=768, num_heads=8,mlp_ratio=4, qkv_bias=False, qkv_scale=None, drop_ratio=0., drop_path_ratio=0., attn_drop_ratio=0.)
    block_feature = blocks(embed)
    print("block feature shape: {}".format(block_feature.shape))