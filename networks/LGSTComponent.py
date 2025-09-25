from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

from mmengine.model import BaseModule

from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.utils import to_2tuple
from typing import Sequence

import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

class PatchMerging(BaseModule):
    """Downsample

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sampler = nn.Conv2d(in_channels, 4*in_channels, kernel_size=kernel_size, stride=2, padding=1)

        sample_dim = 4*in_channels #kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        x = self.sampler(x)

        output_size = x.shape[-2:]
        x = x.flatten(2)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size

class Upsample(BaseModule):
    """Upsample

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sampler = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=1)

        sample_dim = in_channels #kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        x = self.sampler(x)

        output_size = x.shape[-2:]
        x = x.flatten(2)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        # self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, q, k, v, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = q.shape
        q, k, v = map(lambda x: x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3), [q, k, v])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                #  shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def win_reshape(self, query, B, H, W, C):
        query = query.view(B, H, W, C)
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
        else:
            shifted_query = query
        
        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)
            
        return query_windows, H_pad, W_pad, pad_r, pad_b

    def get_mask(self, H_pad, W_pad, device):
        # H_pad, W_pad = query.shape[1], query.shape[2]
        # cyclic shift
        if self.shift_size > 0:           
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                               -float('inf')).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        return attn_mask
    
    def forward(self, query, key, value, hw_shape, shift_size):
        self.shift_size = shift_size

        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query_windows, H_pad, W_pad, pad_r, pad_b = self.win_reshape(query, B, H, W, C)
        key_windows = self.win_reshape(key, B, H, W, C)[0]
        value_windows = self.win_reshape(value, B, H, W, C)[0]
        attn_mask = self.get_mask(H_pad, W_pad, query.device)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, key_windows, value_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

class LSM_SA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size_ls = [0, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
    
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size_ls = shift_size_ls
        self.scale = qk_scale or (embed_dims // num_heads)**-0.5
        
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.win_attn = ShiftWindowMSA(
                                    embed_dims,
                                    num_heads,
                                    window_size,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    attn_drop_rate=attn_drop_rate,
                                    proj_drop_rate=proj_drop_rate,
                                    dropout_layer=dropout_layer,)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        for step in self.shift_size_ls:
            v = self.win_attn(q, k, v, hw_shape = hw_shape, shift_size=step)
        return v
    

class GC_SA(nn.Module):
    def __init__(self, dim, compress_size, compress_overlap, head_num, head_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.compress_conv = nn.Conv2d(dim, 
                                       dim, 
                                       kernel_size=(compress_size, compress_size), 
                                       stride=(compress_size-compress_overlap, compress_size-compress_overlap), 
                                       padding=(0, 0))
        self.dim = dim
        self.head_num = head_num
        self.head_dim = head_dim

    def group_split(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], self.row_num, self.col_num, -1).permute(0, 3, 1, 2)
        x = self.compress_conv(x)
        x = x.reshape(x.shape[0], self.dim, -1).permute(0, 2, 1)
        return x
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, shape_row_col: Union[list, tuple]):
        self.row_num, self.col_num = shape_row_col
        assert self.row_num * self.col_num == q.shape[1], 'input shape should be equal to row_num * col_num'

        k_com = self.group_split(k) 
        v_com = self.group_split(v)
        reshape2bhnd = lambda q: q.reshape(q.shape[0], q.shape[1], self.head_num, self.head_dim).permute(0, 2, 1, 3)
        q, k_com, v_com = map(reshape2bhnd, [q, k_com, v_com])
        out = F.scaled_dot_product_attention(q, k_com, v_com)
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], self.row_num*self.col_num, -1)
        return out
            
class LGSTModule(nn.Module):
    def __init__(self, 
                 dim, 
                 head_num, 
                 head_dim = None, 
                 win_size = 8, 
                 compress_size = 8,
                 stride = [0, 4], 
                 qkv_bias = True, 
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout = 0.):
        super().__init__()
        assert dim % head_num == 0, 'dim must be divided by heads'        
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num if head_dim is None else head_dim
        self.win_size = win_size
        self.stride = stride
        self.scale = self.head_dim ** -0.5
        inner_dim = self.head_dim * self.head_num
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3,  bias=qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.W_gate = nn.Linear(inner_dim, 2)
        
        compress_overlap = 2
        self.GC_SA = GC_SA(dim, compress_size, compress_overlap, head_num = head_num, head_dim = self.head_dim)

        self.input_shape = None
        self.group_mask = None

        self.swat_block = LSM_SA(embed_dims = dim,
                                    num_heads = head_num,
                                    window_size = win_size,
                                    shift_size_ls = [0, 4],
                                    qkv_bias=True,
                                    qk_scale=None,
                                    attn_drop_rate=attn_drop_rate,
                                    proj_drop_rate=proj_drop_rate,
                                    dropout_layer=dict(type='DropPath', drop_prob=0.),
                                    )
    
    def init_shape_params(self, x, shape_row_col):
        self.input_shape = x.shape
        self.row_num, self.col_num = shape_row_col
        assert self.row_num * self.col_num == self.input_shape[1], 'input shape should be equal to row_num * col_num'
        
        W_gate_channel = nn.Linear(self.input_shape[1], 2, device=x.device)
        self.add_module('W_gate_channel', W_gate_channel)
        

     
    def forward(self, x: Tensor, shape_row_col: Union[list, tuple]):
        if self.input_shape != x.shape:
            self.init_shape_params(x, shape_row_col)

        o_win = self.swat_block(x, hw_shape = shape_row_col)
        
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        o_com = self.GC_SA(q, k, v, shape_row_col)
        
        gate = self.W_gate(x).unsqueeze(-2)
        gate_channel = self.W_gate_channel(x.transpose(-1, -2)).unsqueeze(-3)
        weight = F.sigmoid(gate*gate_channel)
        o_win_com = torch.stack([o_win, o_com], dim = -1)

        out = torch.sum(o_win_com * weight, dim=-1) 
        return self.to_out(out)

if __name__ == '__main__':

    #  test module
    x = torch.randn(8, 128*128, 96).to('cuda')
    hw_shape = (128, 128)
    lgst_module = LGSTModule(dim=96, head_num=3, win_size=8, stride=[0, 4]).to('cuda')
    out = lgst_module(x, hw_shape)
    print(out.shape)
    swat_block = LSM_SA(embed_dims=96, num_heads=3, window_size=8, shift_size_ls=[0, 4]).to('cuda')
    out =  swat_block(x, hw_shape)
    print(out.shape)

