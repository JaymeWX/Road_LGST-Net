import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from functools import partial
from networks.common import DecoderBlock
from einops import rearrange, repeat

nonlinearity = partial(F.relu, inplace=True)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        qk_scale=None,
        dropout = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        # torch.Tensor.__matmul__()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        x = self.dropout(x)
        return x

# Because the paper has not been accepted, Our team felt there was some risk in giving away the complete code.
# Therefore, this core module of our project has been hidden temporarily.
# If the paper is accepted, we will release the code at the first time.
class ShiftWindAttention(nn.Module):
    def __init__(self, dim, heads, dim_head = 64, dropout = 0., win_size = 8, stride = [0, 4], patch_w_num = 32, qkv_bias = False):
        pass

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



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
        patch_w_num = 0,
        win_size = 8,
        stride = [0, 4],
        has_bn = False #has batchnorm
    ):
        super().__init__()
        self.is_swat = is_SWAT
        # self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.has_bn = has_bn
        if is_SWAT:
            self.attn = ShiftWindAttention(
                dim,
                heads = num_heads,
                qkv_bias = qkv_bias,
                patch_w_num = patch_w_num,
                dropout = dropout,
                win_size = win_size,
                stride = stride
            )
        else:
            self.attn = Attention(
                dim,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
            )
        # self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )
        self.dropout = nn.Dropout(dropout)
        if has_bn: self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        if self.is_swat == False:
            W = x.shape[1]
            x = rearrange(x, 'B W H C -> B (W H) C')

        x = x + self.attn(x)
        x = self.dropout(x)
        x = x + self.mlp(x)
        
        if self.is_swat == False:
            x = rearrange(x, 'B (W H) C -> B W H C', W = W)

        
        if self.has_bn:
            x = rearrange(x, 'B W H C -> B C W H')
            x = self.batch_norm(x)
            x = rearrange(x, 'B C W H -> B W H C')

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

SCALE_LIST = ['light', 'normal']
class SWATNet(nn.Module):
    def __init__(self, num_classes=1, is_swat = True, scale = SCALE_LIST[0]):
        super(SWATNet, self).__init__()

        filters = [64, 128, 256, 512]
        self.is_swat = is_swat
        img_size, patch_size, in_chans, patch_embed_dim = 512, 4, 3, filters[0]
        self.patch_layer = nn.Sequential(PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim),
                                      nn.BatchNorm2d(filters[0]))       
        
        self.creat_vit_blocks(scale)

        self.downsample1 = BasicBlock(filters[0], filters[1])
        self.downsample2 = BasicBlock(filters[1], filters[2])
        self.downsample3 = BasicBlock(filters[2], filters[3])

        self.upsample3 = DecoderBlock(filters[3], filters[2])
        self.upsample2 = DecoderBlock(filters[2], filters[1])
        self.upsample1 = DecoderBlock(filters[1], filters[0])

        self.finalupsample = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(16, num_classes, 3, padding=1)
        

    def creat_vit_blocks(self, scale = SCALE_LIST[0]):
        if scale == SCALE_LIST[0]:
            self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = self.is_swat)
            self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)

            self.vit_block_d3 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)
            self.vit_block_d2 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_d1 = Block(64, 2, patch_w_num = 128, is_SWAT = self.is_swat)
        elif scale == SCALE_LIST[1]:
            block_e1 = [Block(64, 2, patch_w_num = 128, is_SWAT = True) for i in range(1)]
            block_e2 = [Block(128, 4, patch_w_num = 64, is_SWAT = True) for i in range(2)]
            block_e3 = [Block(256, 8, patch_w_num = 32, is_SWAT = True) for i in range(3)]

            block_d3 = [Block(256, 8, patch_w_num = 32, is_SWAT = True) for i in range(3)]
            block_d2 = [Block(128, 4, patch_w_num = 64, is_SWAT = True) for i in range(2)]
            block_d1 = [Block(64, 2, patch_w_num = 128, is_SWAT = True) for i in range(1)]
            
            self.vit_block_e1 = nn.ModuleList(block_e1)
            self.vit_block_e2 = nn.ModuleList(block_e2)
            self.vit_block_e3 = nn.ModuleList(block_e3)
            self.vit_block_d3 = nn.ModuleList(block_d3)
            self.vit_block_d2 = nn.ModuleList(block_d2)
            self.vit_block_d1 = nn.ModuleList(block_d1)
            
    
    def vit_ops(self, x, vit_block):
        
        x = rearrange(x, 'B C H W -> B H W C')
        if isinstance(vit_block, nn.ModuleList):
            for index, block in enumerate(vit_block):
                x = block(x)
        else:
            x = vit_block(x)
        x = rearrange(x, 'B H W C -> B C H W')
        return x

    def forward(self, x):
        
        # input to patch embedding
        e1 = self.patch_layer(x)  # H/4， W/4

        # Encoder
        e1 = self.vit_ops(e1, self.vit_block_e1) # H/4， W/4
        e2 = self.downsample1(e1) 
        e2 = self.vit_ops(e2, self.vit_block_e2) # H/8， W/8
        e3 = self.downsample2(e2)
        e3 = self.vit_ops(e3, self.vit_block_e3) # H/16， W/16
        e4 = self.downsample3(e3)  # H/32， W/32

        # Decoder
        d4 = self.upsample3(e4) + e3
        d4 = self.vit_ops(d4, self.vit_block_d3) 
        d3 = self.upsample2(d4) + e2
        d3 = self.vit_ops(d3, self.vit_block_d2)
        d2 = self.upsample1(d3) + e1
        d2 = self.vit_ops(d2, self.vit_block_d1)

        # output head
        d1 = self.finalupsample(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
