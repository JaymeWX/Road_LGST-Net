import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from functools import partial
from networks.common import DecoderBlock
from networks.SWATNet import ShiftWindAttention, Attention, Mlp, PatchEmbed
from einops import rearrange



nonlinearity = partial(F.relu, inplace=True)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.ReLU(inplace=True),
        dropout = 0.0,
        is_SWAT = True,
        patch_w_num = 0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if is_SWAT:
            self.attn = ShiftWindAttention(
                dim,
                heads = num_heads,
                qkv_bias = qkv_bias,
                patch_w_num = patch_w_num,
                dropout = dropout
            )
        else:
            self.attn = Attention(
                dim,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
            )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.dropout(x)
        x = x + self.mlp(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nonlinearity
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        return x

class SWATNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, is_swat = True):
        super(SWATNet, self).__init__()

        filters = [64, 128, 256, 512]
        
        img_size, patch_size, in_chans, patch_embed_dim = 512, 4, 3, filters[0]
        self.encoder1 = nn.Sequential(PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim),
                                      nn.BatchNorm2d(filters[0]))       
        
        self.encoder2 = BasicBlock(filters[0], filters[1])
        self.encoder3 = BasicBlock(filters[1], filters[2])
        self.encoder4 = BasicBlock(filters[2], filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(16, num_classes, 3, padding=1)

        self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)
        self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
        self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)

        self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
        self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
        self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)

    def vit_ops(self, x, vit_block):
        
        x = rearrange(x, 'B C H W -> B H W C')
        x = vit_block(x)
        x = rearrange(x, 'B H W C -> B C H W')
        return x

    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)

        e1 = self.encoder1(x)  # H/4， W/4
        e1 = self.vit_ops(e1, self.vit_block_e1) # H/4， W/4

        e2 = self.encoder2(e1) 
        e2 = self.vit_ops(e2, self.vit_block_e2) # H/8， W/8

        e3 = self.encoder3(e2)
        e3 = self.vit_ops(e3, self.vit_block_e3) # H/16， W/16

        e4 = self.encoder4(e3)  # H/32， W/32

        # Center
        # Decoder
        d4 = self.decoder4(e4) + e3
        d4 = self.vit_ops(d4, self.vit_block_d4) 

        d3 = self.decoder3(d4) + e2
        d3 = self.vit_ops(d3, self.vit_block_d3)

        d2 = self.decoder2(d3) + e1
        d2 = self.vit_ops(d2, self.vit_block_d2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
