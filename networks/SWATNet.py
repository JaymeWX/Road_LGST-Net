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


class ShiftWindAttention(nn.Module):
    def __init__(self, dim, heads, dim_head = 64, dropout = 0., win_size = 8, stride = [0, 4], patch_w_num = 32, qkv_bias = False):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.patch_w_num = patch_w_num
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
 
        self.to_qkv = nn.Linear(dim, inner_dim * 3,  bias=qkv_bias)
        # self.qkv_pro = nn.Sequential(nn.GELU(), nn.Linear(inner_dim * 3, inner_dim * 3))
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        masks = []
        for step in stride:
            masks.append(self.get_mask(win_size, step))
        masks = torch.stack(masks)
        self.register_buffer("attn_masks", masks)
        self.init_position_param(win_size)
 
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # qkv = self.qkv_pro(qkv).chunk(3, dim = -1)
        # d = c * h
        # q, k, v = map(lambda t: rearrange(t, 'b (ph pw) d -> b ph pw d', pw = self.patch_w_num), qkv)
        out_list = []
        for index, shift_step in enumerate(self.stride):
            out = self.cal_window(q, k, v, self.win_size, shift_step, self.attn_masks[index])
            out_list.append(out)

        out = torch.stack(out_list, dim = -1).mean(dim = -1)
        # out = torch.cat(out_list, dim = -1)
        return self.to_out(out)


    def cal_window(self, q, k, v, win_size, stride, mask):
        BS, H, W, _ = q.shape #BS is batch size
        win_num = (H//win_size)**2
        qs = self.window_partition(q, win_size, stride)
        ks = self.window_partition(k, win_size, stride)
        vs = self.window_partition(v, win_size, stride)
        nWb = qs.shape[0]
        qs, ks, vs = map(lambda t: rearrange(t, 'nWb wh ww (h c) -> nWb h (wh ww) c', h = self.heads), [qs, ks, vs])
        
        dots = torch.matmul(qs, ks.transpose(-1, -2)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            win_size * win_size, win_size * win_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        dots = dots + repeat(relative_position_bias, 'h c1 c2 -> nWb h c1 c2', nWb = nWb)

        dots = dots + repeat(mask, 'wn c1 c2 -> (bs wn) h c1 c2', bs = BS, h = self.heads)
        
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vs)
        out = self.window_reverse(out, win_size, H, W, stride)
        # out = rearrange(out, 'b h H W c -> b (H W) (h c)')
        return out

    def get_mask(self, window_size, shift_size):
        H, W = pair(self.patch_w_num)
        if shift_size > 0:
            # calculate attention mask for SWA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = torch.zeros(((H//window_size)**2, window_size * window_size, window_size * window_size))

        return attn_mask
    
    def window_partition(self, x, window_size, shift = 0):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2)) 

        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W, shift = 0):
        """
        Args:
            windows: (num_windows*B, h, window_size*window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = rearrange(windows, '(b nWh nWw) h (wh ww) c -> b (nWh wh) (nWw ww) (h c)', b = B, nWh = H // window_size, wh = window_size)

        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))
        return x
    
    def init_position_param(self, win_size):
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), self.heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(win_size)
        coords_w = torch.arange(win_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += win_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += win_size - 1
        relative_coords[:, :, 0] *= 2 * win_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


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

SCALE_LIST = ['light', 'middle', 'large']
class SWATNet(nn.Module):
    def __init__(self, num_classes=1, is_swat = True, scale = SCALE_LIST[0]):
        super(SWATNet, self).__init__()

        filters = [64, 128, 256, 512]
        self.is_swat = is_swat
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
        self.creat_vit_blocks(scale)
        

    def creat_vit_blocks(self, scale = SCALE_LIST[0]):
        repeat_block = lambda x, n : [x for _ in range(n)]
        if scale == SCALE_LIST[0]:
            self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = self.is_swat)
            self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)

            self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)
            self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = self.is_swat)
        elif scale =='v2':
            self.vit_block_e1 = Block(64, 2, patch_w_num = 128, stride=[0, 4, 8, 12], is_SWAT = self.is_swat)
            self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)

            self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)
            self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_d2 = Block(64, 2, patch_w_num = 128, stride=[0, 4, 8, 12], is_SWAT = self.is_swat)
        elif scale == 'v3':
            self.vit_block_e1 = Block(64, 2, patch_w_num = 128, win_size=16, stride=[0, 8], is_SWAT = self.is_swat)
            self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)

            self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = self.is_swat)
            self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = self.is_swat)
            self.vit_block_d2 = Block(64, 2, patch_w_num = 128, win_size=16, stride=[0, 8], is_SWAT = self.is_swat)
        elif scale == 'vit':
            block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = True)
            block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = True)
            block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = False)

            block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = False)
            block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = True)
            block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = True)
            
            self.vit_block_e1 = nn.Sequential(*repeat_block(block_e1,1))
            self.vit_block_e2 = nn.Sequential(*repeat_block(block_e2,4))
            self.vit_block_e3 = nn.Sequential(*repeat_block(block_e3,6))
            self.vit_block_d4 = nn.Sequential(*repeat_block(block_d4,6))
            self.vit_block_d3 = nn.Sequential(*repeat_block(block_d3,4))
            self.vit_block_d2 = nn.Sequential(*repeat_block(block_d2,1))
        elif scale == 'v4':
            block_e1 = [Block(64, 2, patch_w_num = 128, is_SWAT = True) for i in range(1)]
            block_e2 = [Block(128, 4, patch_w_num = 64, is_SWAT = True) for i in range(2)]
            block_e3 = [Block(256, 8, patch_w_num = 32, is_SWAT = True) for i in range(3)]

            block_d4 = [Block(256, 8, patch_w_num = 32, is_SWAT = True) for i in range(3)]
            block_d3 = [Block(128, 4, patch_w_num = 64, is_SWAT = True) for i in range(2)]
            block_d2 = [Block(64, 2, patch_w_num = 128, is_SWAT = True) for i in range(1)]
            
            self.vit_block_e1 = nn.ModuleList(block_e1)
            self.vit_block_e2 = nn.ModuleList(block_e2)
            self.vit_block_e3 = nn.ModuleList(block_e3)
            self.vit_block_d4 = nn.ModuleList(block_d4)
            self.vit_block_d3 = nn.ModuleList(block_d3)
            self.vit_block_d2 = nn.ModuleList(block_d2)
        elif scale == 'v5':
            block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = True, has_bn=True)
            block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = True, has_bn=True)
            block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = True, has_bn=True)

            block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = True, has_bn=True)
            block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = True, has_bn=True)
            block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = True, has_bn=True)
            
            self.vit_block_e1 = nn.Sequential(*repeat_block(block_e1,1))
            self.vit_block_e2 = nn.ModuleList(repeat_block(block_e2,4))
            self.vit_block_e3 = nn.ModuleList(repeat_block(block_e3,4))
            self.vit_block_d4 = nn.ModuleList(repeat_block(block_d4,4))
            self.vit_block_d3 = nn.ModuleList(repeat_block(block_d3,4))
            self.vit_block_d2 = nn.Sequential(*repeat_block(block_d2,1))
        elif scale == 'no_attention':
            self.vit_block_e1 = nn.Identity()
            self.vit_block_e2 = nn.Identity()
            self.vit_block_e3 = nn.Identity()
            self.vit_block_d4 = nn.Identity()
            self.vit_block_d3 = nn.Identity()
            self.vit_block_d2 = nn.Identity()
            
    
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
