import torch
from torch import nn
import torch.nn.functional as F
 
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math
from typing import List
from functools import partial
from networks.dlinknet import Dblock
from networks.common import nonlinearity, DecoderBlock
from torchvision import models
from timm.models.layers import trunc_normal_, DropPath# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)




class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# classes 
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
 
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
 
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
 
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        # self.to_v = nn.Linear(dim, inner_dim, bias = False)
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q = self.to_qk(x)
        # k = self.to_qk(x)
        # v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
 
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
 
        attn = self.attend(dots)
        attn = self.dropout(attn)
 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class WindowAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., win_size = 8, stride = [0, 4], patch_w_num = 32, qkv_bias = False):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.patch_w_num = patch_w_num
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
 
        self.to_qkv = nn.Linear(dim, inner_dim * 3,  bias=qkv_bias)
        self.qkv_pro = nn.Sequential(nn.GELU(), nn.Linear(inner_dim * 3, inner_dim * 3))
 
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
        qkv = self.to_qkv(x)
        qkv = self.qkv_pro(qkv).chunk(3, dim = -1)
        # d = c * h
        q, k, v = map(lambda t: rearrange(t, 'b (ph pw) d -> b ph pw d', pw = self.patch_w_num), qkv)
        out_list = []
        for index, shift_step in enumerate(self.stride):
            out = self.cal_window(q, k, v, self.win_size, shift_step, self.attn_masks[index])
            out_list.append(out)
        out = torch.stack(out_list, dim = -1).mean(dim = -1)
        # out = reduce(out, 'b n d c -> b n d', 'mean')

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # attn = self.dropout(attn)
 
        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


    def cal_window(self, q, k, v, win_size, stride, mask):
        BS, H, W, _ = q.shape #BH is batch size
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
        # out = rearrange(out, 'nWb h (wsh wsw) c-> nWb wsh wsw c', wsh = win_size)
        out = self.window_reverse(out, win_size, H, W, stride)
        out = rearrange(out, 'b h H W c -> b (H W) (h c)')
        return out

    def get_mask(self, window_size, shift_size):
        H, W = pair(self.patch_w_num)
        if shift_size > 0:
            # calculate attention mask for SW-MSA
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
        # x = windows.view(B, H // window_size, W // window_size, self.heads, window_size, window_size, -1)
        # x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, self.heads, H, W, -1)
        x = rearrange(windows, '(b nWh nWw) h (wh ww) c -> b h (nWh wh) (nWw ww) c', b = B, nWh = H // window_size, wh = window_size)

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

        

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
 
        # self.to_kv = nn.Linear(dim, inner_dim, bias = False)
        self.q_project = nn.Linear(dim, inner_dim, bias = False)
        self.k_project = nn.Linear(dim, inner_dim, bias = False)
        self.v_project = nn.Linear(dim, inner_dim, bias = False)
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, q, k, v = None): 
        v = k if v is None else v
        q = self.q_project(q)
        k = self.k_project(k)
        v = self.v_project(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])
 
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
 
        attn = self.attend(dots)
        attn = self.dropout(attn)
 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., out_indices = (9, 14, 19, 23), patch_w_num = 32):
        super().__init__()
        
        if out_indices == -1:
            self.out_indices = [depth - 1]
        else:
            self.out_indices = out_indices
        assert self.out_indices[-1] == depth - 1
 
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, WindowAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_w_num=patch_w_num)),
                # PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
 
    def forward(self, x):
        out = []
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
 
            if index in self.out_indices:
                out.append(x)
 
        return out


def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        # print('position embeding sum:', torch.sum(abs_pos))
        return abs_pos


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., out_indices = (9, 14, 19, 23)):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_w_num =int(image_height / patch_height)
        num_patches =int((image_height // patch_height) * (image_width // patch_width))
        patch_dim = channels * patch_height * patch_width
 
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
 
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        trunc_normal_(self.pos_embed, std=.02)


        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices, patch_w_num = patch_w_num)
        self.out = Rearrange("b (h w) c->b c h w", w = patch_w_num)
 
    def forward(self, img, reshape_out = True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        w = h = int(math.sqrt(n))
        x +=  get_abs_pos(self.pos_embed, False, [h, w])
        x = self.dropout(x)
 
        out = self.transformer(x)
 
        for index, transformer_out in enumerate(out):
            # delete cls_tokens and transform output to [b, c, h, w]
            if reshape_out == True:
                out[index] = self.out(transformer_out)
            else:
                out[index] = transformer_out
        if len(out) == 1:
            out = out[0]
        return out
 

class PUPHead(nn.Module):
    def __init__(self, num_classes, input_dim = 1024):
        super(PUPHead, self).__init__()
        dim_list = [input_dim, 256, 128, 64, 32]
        
        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(dim_list[0], dim_list[1], 3, padding=1),
            nn.BatchNorm2d(dim_list[1]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(dim_list[1], dim_list[2], 3, padding=1),
            nn.BatchNorm2d(dim_list[2]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_3= nn.Sequential(
            nn.Conv2d(dim_list[2], dim_list[3], 3, padding=1),
            nn.BatchNorm2d(dim_list[3]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_4= nn.Sequential(
            nn.Conv2d(dim_list[3], dim_list[4], 3, padding=1),
            nn.BatchNorm2d(dim_list[4]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
    
        self.cls_seg = nn.Conv2d(dim_list[4], num_classes, 3, padding=1)
 
    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        x = self.UP_stage_4(x)
        x = self.cls_seg(x)
        return x

class ConvTransHead(nn.Module):
    def __init__(self):
        super().__init__()
        output_dim_after_upscaling = 256
        self.output_layers = nn.ModuleList([])
        upscaling_layer_dims = [128, 64, 32, 16]
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.output_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims),
                    # if idx < len(upscaling_layer_dims) - 1
                    # else nn.Identity(),
                    nn.ReLU()
                    # if idx < len(upscaling_layer_dims) - 1
                    # else nn.Identity(),
                )
            )
            output_dim_after_upscaling = layer_dims
        self.last_conv = nn.Conv2d(upscaling_layer_dims[-1], 1, 3, padding = 1)

    def forward(self, x):
        for ids, output_layer in enumerate(self.output_layers):
            x = output_layer(x)
        x = self.last_conv(x)
        return x

class ViTRoad(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(ViTRoad, self).__init__()

        # filters = [64, 128, 256, 512]
        filters = [8, 16, 32, 64]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
  
        self.encoder = ViT(image_size=512, patch_size=4, dim=64, depth=4, heads=4, 
                           mlp_dim=128, channels=64, dim_head=32, out_indices=(0, 1, 2, 3))
        # self.neckblock = Transformer(512, 4, 8, 128, 1024, 0, -1, 32)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[1], 8, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(8, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        e1, e2, e3, e4 = self.encoder(x)
        e4 = e1 + e2 + e3 + e4 #torch.Size([8, 64, 128, 128])

        # Decoder
        d4 = self.decoder4(e4) 
        d3 = self.decoder3(d4)
        # d2 = self.decoder2(d3)
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d3)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)