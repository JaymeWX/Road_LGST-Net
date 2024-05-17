# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .windvitnet import ShiftWindAttention


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
        out = self.window_reverse(out, win_size, H, W, stride)
        out = rearrange(out, 'b h H W c -> b (H W) (h c)')
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
        self.act = act_layer()
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
        act_layer=nn.GELU,
        dropout = 0.05,
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
                patch_w_num = patch_w_num
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
        x = x + self.attn(self.norm1(x))
        x = self.dropout(x)
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
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
        return abs_pos.reshape(1, h, w, -1)


# Image encoder 
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
        is_SWAT: bool = True,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True, patch_w_num=self.image_embedding_size, is_SWAT=is_SWAT)
            self.blocks.append(vit_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


# Image decoder for road detection.
class ImageDecoderViT(nn.Module):
    def __init__(
        self,
        patch_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer : nn.Module,
        patch_w_num,
        is_SWAT = True
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        upscaling_layer_dims = [128, 64, 32, 16]

        # Initialize absolute positional embedding with pretrain image size.
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True, act_layer = act_layer, patch_w_num = patch_w_num, is_SWAT=is_SWAT)
            self.blocks.append(vit_block)
        

        output_dim_after_upscaling = 256
        self.output_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.output_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    act_layer() 
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.head = nn.Sequential(Mlp(16,8,1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, batched_images: torch.Tensor) -> torch.Tensor:   #c*256*64*64
        x = x.permute(0, 2, 3, 1)
        num_patches = x.shape[1]
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        for index, blk in enumerate(self.blocks):
            x = blk(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], num_patches, num_patches)

        for ids, output_layer in enumerate(self.output_layers):
            x = output_layer(x)

        b, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1],  x.shape[2]* x.shape[3])
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], h, w)
        return x


class SWATNet(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        mask_decoder: ImageDecoderViT,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )


    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
        self,
        batched_images: torch.Tensor,
        # scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch_size, _, input_h, input_w = batched_images.shape
        # with torch.no_grad():
        image_embeddings = self.get_image_embeddings(batched_images)
        mask = self.mask_decoder(image_embeddings, batched_images)
        return mask

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_road_SWATNet(encoder_patch_embed_dim, encoder_num_heads, img_size = 1024, encoder_depth = 12, decoder_depth = 12, checkpoint=None):
    encoder_patch_size = 16
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    patch_w_num = img_size//encoder_patch_size
    activation = "gelu"
    normalization_type = "layer_norm"

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        image_encoder.load_state_dict(state_dict)

        
    image_decoder = ImageDecoderViT(
        patch_embed_dim=encoder_neck_dims[0],
        depth=decoder_depth,
        num_heads=4,
        mlp_ratio=encoder_mlp_ratio,
        act_layer=activation_fn,
        patch_w_num = patch_w_num
    )

    swat_net = SWATNet(
        image_encoder=image_encoder,
        mask_decoder=image_decoder,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    
    return swat_net
