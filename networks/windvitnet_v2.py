import torch
from torch import nn
import torch.nn.functional as F
 
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math
from functools import partial
from networks.dlinknet import Dblock
from networks.common import nonlinearity, DecoderBlock
# helpers
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., win_size = 8, stride = [0, 4], patch_w_num = 32):
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
 
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
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


# class WindowAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., win_size = 8, stride = 4, patch_w_num = 32):
#         super().__init__()
#         self.win_size = win_size
#         self.stride = stride
#         self.patch_w_num = patch_w_num
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
 
#         self.heads = heads
#         self.scale = dim_head ** -0.5
 
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)
 
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.qkv_pro = nn.Sequential(nn.GELU(), nn.Linear(inner_dim * 3, inner_dim * 3))
 
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
 
#     def forward(self, x):
#         qkv = self.to_qkv(x)
#         qkv = self.qkv_pro(qkv).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#         out_list = []
#         for stride in range(0, self.win_size, self.stride):
#             out = self.cal_window(q, k, v, self.win_size, stride)
#             out_list.append(out)
#         out = torch.stack(out_list, dim = -1)
#         out = reduce(out, 'b n d c -> b n d', 'mean')

#         # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         # attn = self.attend(dots)
#         # attn = self.dropout(attn)
 
#         # out = torch.matmul(attn, v)
#         # out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

#     def cal_window(self, q, k, v, win_size, stride):
#         range_q, qs = self.window_split(q, win_size=win_size, stride = stride)
#         range_k, ks = self.window_split(k, win_size=win_size, stride = stride)
#         range_v, vs = self.window_split(v, win_size=win_size, stride = stride)
#         # vs_out = []
#         q_, k_, v_ = map(lambda t: torch.stack(t, dim = 0), [qs, ks, vs])
#         # for q_, k_, v_ in zip(qs, ks, vs):
#         #     if q_ == None or k_ == None or v_ == None:
#         #         vs_out.append(None)
#         #         continue
#         dots = torch.matmul(q_, k_.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#         out = torch.matmul(attn, v_)
#         # vs_out.append(out)
        
#         out = self.window_merge(range_v, out)
#         # print(out.is_contiguous())
#         out = rearrange(out, 'b h num_h num_w d -> b (num_h num_w) (h d)')
#         return out

#     def window_split(self, x, win_size = 8, stride = 2):
#         w = self.patch_w_num
#         x = rearrange(x, 'b h (num_h num_w) d -> b h num_h num_w d', num_h = w)
#         assert w % win_size == 0 
#         assert stride < win_size
#         i = [n for n in range(-win_size, w+1, win_size)]
#         j = [n for n in range(-win_size, w+1, win_size)]
#         h_range = list(zip(i[:-1], i[1:]))
#         w_range = list(zip(j[:-1], j[1:]))
#         block_ranges = [[i[0], j[0], i[1], j[1]] for i in h_range for j in w_range]
#         ranges = []
#         for r in block_ranges:
#             temp = r
#             for ids, num in enumerate(r):
#                 num = num + stride
#                 num = min(max(num, 0), w)
#                 temp[ids] = num
#             if (temp[2] != temp[0]) and (temp[3] != temp[1]):
#             #      temp = [0, 0, 0, 0]
#             # else:
#                 ranges.append(temp)
#         blocks = []
#         for range_ in ranges:
#             if range_[0] == range_[2] or range_[3] == range_[1]:
#                 blocks.append(None)
#             else:
#                 b = x[:, :, range_[0]:range_[2], range_[1]:range_[3], :]
#                 b = rearrange(b, 'b h num_h num_w d -> b h (num_h num_w) d')
#                 shape = b.size()
#                 b = F.pad(b, (0, 0, 0, win_size**2 - shape[-2]), 'constant', 0)
#                 blocks.append(b)
#         return ranges, blocks
    
#     def window_merge(self, block_ranges, blocks):
#         h_count = int(math.sqrt(len(block_ranges)))
#         assert h_count**2 == len(block_ranges)
#         col_list = []
#         for i in range(0, len(block_ranges), h_count):
#             row_list = []
#             for j in range(h_count):
#                 range_ = block_ranges[i+j]
#                 if range_[0] != range_[2] and range_[3] != range_[1]:
#                     num = (range_[2] - range_[0])*(range_[3] - range_[1])
#                     block = blocks[i+j, :, :, :num, :]
#                     block = rearrange(block, 'b h (num_h num_w) d -> b h num_h num_w d', num_h = range_[2] - range_[0])
#                     row_list.append(block)
#             if len(row_list) != 0:
#                 row = torch.cat(row_list, dim = -2)
#                 col_list.append(row)
#         if len(col_list) != 0:
#             res = torch.cat(col_list, dim = -3)
#             return res
#         else:
#             return None
        # b, h, num, d = blocks[0].shape
        # x = torch.Tensor((b, h, w, w, d), device = blocks[0].device)
        # for range_, block in zip(block_ranges, blocks):
        #     if range_[0] != range_[2] and range_[3] != range_[1]:
        #         block = rearrange(block, 'b h (num_h num_w) d -> b h num_h num_w d', num_h = range_[2] - range_[0])
        #         x[:, :, range_[0]:range_[2], range_[1]:range_[3], :] = block
        

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
 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices, patch_w_num = patch_w_num)
        self.out = Rearrange("b (h w) c->b c h w", w = patch_w_num)
 
    def forward(self, img, reshape_out = True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :]
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
    def __init__(self, image_size, num_classes = 1, dropout = 0., emb_dropout = 0.):
        super(ViTRoad, self).__init__()
        self.num_classes = num_classes
        # self.patch_size = patch_size
        # self.patch_h_num, self.patch_w_num = pair(int(image_size//patch_size))
        head_num = 8
        encoder_layer = partial(ViT, patch_size=2, depth=1, heads=head_num, dropout = dropout, emb_dropout = emb_dropout, out_indices = -1)

        # en_dim, en_depth, en_heads, en_mlp_dim, en_dim_head = [dim, depth, heads, mlp_dim, dim_head]
        # de_dim, de_depth, de_heads, de_mlp_dim, de_dim_head = [dim, depth, heads, mlp_dim, dim_head]
        neck_dims = [256, 128, 256]
        dim_list = [64, 128, 256, 512]
        # channel_list = []

        self.first_conv_layer = nn.Conv2d(3, 64, 7, 2, 3)        
        self.encoder1 = encoder_layer(image_size=image_size/2, dim=dim_list[0], mlp_dim=2*dim_list[0], channels = 64, dim_head = int(dim_list[0]/head_num))
        self.encoder2 = encoder_layer(image_size=image_size/4, dim=dim_list[1], mlp_dim=2*dim_list[1], channels = dim_list[0], dim_head = int(dim_list[1]/head_num))
        self.encoder3 = encoder_layer(image_size=image_size/8, dim=dim_list[2], mlp_dim=2*dim_list[2], channels = dim_list[1], dim_head = int(dim_list[2]/head_num))
        self.encoder4 = encoder_layer(image_size=image_size/16, dim=dim_list[3], mlp_dim=2*dim_list[3], channels = dim_list[2], dim_head = int(dim_list[3]/head_num))

        # self.encoder3 = Transformer(dim=)
        # self.encoder4 = Transformer(dim=)
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(dim_list[3], dim_list[2])
        self.decoder3 = DecoderBlock(dim_list[2], dim_list[1])
        self.decoder2 = DecoderBlock(dim_list[1], dim_list[0])
        self.decoder1 = DecoderBlock(dim_list[0], dim_list[0])

        self.finaldeconv1 = nn.ConvTranspose2d(dim_list[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        
    def forward(self, x):
        #encoder
        x = self.first_conv_layer(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        #neck
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
