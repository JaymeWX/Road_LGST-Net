import torch
from torch import nn
import torch.nn.functional as F
 
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., win_size = 8, stride = 4, patch_w_num = 32):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.patch_w_num = patch_w_num
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
 
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.qkv_pro = nn.Sequential(nn.GELU(), nn.Linear(inner_dim * 3, inner_dim * 3))
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = self.qkv_pro(qkv).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        out_list = []
        for stride in range(0, self.win_size, self.stride):
            out = self.cal_window(q, k, v, self.win_size, stride)
            out_list.append(out)
        out = torch.stack(out_list, dim = -1)
        out = reduce(out, 'b n d c -> b n d', 'mean')

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # attn = self.dropout(attn)
 
        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def cal_window(self, q, k, v, win_size, stride):
        range_q, qs = self.window_split(q, win_size=win_size, stride = stride)
        range_k, ks = self.window_split(k, win_size=win_size, stride = stride)
        range_v, vs = self.window_split(v, win_size=win_size, stride = stride)
        # vs_out = []
        q_, k_, v_ = map(lambda t: torch.stack(t, dim = 0), [qs, ks, vs])
        # for q_, k_, v_ in zip(qs, ks, vs):
        #     if q_ == None or k_ == None or v_ == None:
        #         vs_out.append(None)
        #         continue
        dots = torch.matmul(q_, k_.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_)
        # vs_out.append(out)
        
        out = self.window_merge(range_v, out)
        # print(out.is_contiguous())
        out = rearrange(out, 'b h num_h num_w d -> b (num_h num_w) (h d)')
        return out

    def window_split(self, x, win_size = 8, stride = 2):
        w = self.patch_w_num
        x = rearrange(x, 'b h (num_h num_w) d -> b h num_h num_w d', num_h = w)
        assert w % win_size == 0 
        assert stride < win_size
        i = [n for n in range(-win_size, w+1, win_size)]
        j = [n for n in range(-win_size, w+1, win_size)]
        h_range = list(zip(i[:-1], i[1:]))
        w_range = list(zip(j[:-1], j[1:]))
        block_ranges = [[i[0], j[0], i[1], j[1]] for i in h_range for j in w_range]
        ranges = []
        for r in block_ranges:
            temp = r
            for ids, num in enumerate(r):
                num = num + stride
                num = min(max(num, 0), w)
                temp[ids] = num
            if (temp[2] != temp[0]) and (temp[3] != temp[1]):
            #      temp = [0, 0, 0, 0]
            # else:
                ranges.append(temp)
        blocks = []
        for range_ in ranges:
            if range_[0] == range_[2] or range_[3] == range_[1]:
                blocks.append(None)
            else:
                b = x[:, :, range_[0]:range_[2], range_[1]:range_[3], :]
                b = rearrange(b, 'b h num_h num_w d -> b h (num_h num_w) d')
                shape = b.size()
                b = F.pad(b, (0, 0, 0, win_size**2 - shape[-2]), 'constant', 0)
                blocks.append(b)
        return ranges, blocks
    
    def window_merge(self, block_ranges, blocks):
        h_count = int(math.sqrt(len(block_ranges)))
        assert h_count**2 == len(block_ranges)
        col_list = []
        for i in range(0, len(block_ranges), h_count):
            row_list = []
            for j in range(h_count):
                range_ = block_ranges[i+j]
                if range_[0] != range_[2] and range_[3] != range_[1]:
                    num = (range_[2] - range_[0])*(range_[3] - range_[1])
                    block = blocks[i+j, :, :, :num, :]
                    block = rearrange(block, 'b h (num_h num_w) d -> b h num_h num_w d', num_h = range_[2] - range_[0])
                    row_list.append(block)
            if len(row_list) != 0:
                row = torch.cat(row_list, dim = -2)
                col_list.append(row)
        if len(col_list) != 0:
            res = torch.cat(col_list, dim = -3)
            return res
        else:
            return None
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

# class CrossAttention(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(CrossAttention, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.query = nn.Linear(in_dim, out_dim, bias=False)
#         self.key = nn.Linear(in_dim, out_dim, bias=False)
#         self.value = nn.Linear(in_dim, out_dim, bias=False)

#     def forward(self, q, k, v = None): 
#         batch_size = q.shape[0]
#         num_queries = q.shape[1]
#         num_keys = k.shape[1]

#         v = k if v is None else v
#         q = self.query(q)
#         k = self.key(k)
#         v = self.value(v)

#         # 计算注意力分数
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         # 计算加权和
#         output = torch.bmm(attn_weights, v)
        
#         return output


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., out_indices = (9, 14, 19, 23)):
        super().__init__()
        
        if out_indices == -1:
            self.out_indices = [depth - 1]
        else:
            self.out_indices = out_indices
        assert self.out_indices[-1] == depth - 1
 
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, WindowAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
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
 
        num_patches =int((image_height // patch_height) * (image_width // patch_width))
        patch_dim = channels * patch_height * patch_width
 
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices)
        self.out = Rearrange("b (h w) c->b c h w", h=image_height//patch_height, w=image_width//patch_width)
 
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
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes = 1, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., out_indices = -1):
        super(ViTRoad, self).__init__()
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.patch_h_num, self.patch_w_num = pair(int(image_size//patch_size))

        en_dim, en_depth, en_heads, en_mlp_dim, en_dim_head = [dim, depth, heads, mlp_dim, dim_head]
        de_dim, de_depth, de_heads, de_mlp_dim, de_dim_head = [dim, depth, heads, mlp_dim, dim_head]
        neck_dims = [256, 128, 256]
        out_indices = [0, 1, 2, 3]
        self.encoder = ViT(image_size=image_size, patch_size=patch_size, dim=en_dim, depth=en_depth, heads=en_heads, mlp_dim=en_mlp_dim, 
                        channels = channels, dim_head = en_dim_head, dropout = dropout, emb_dropout = emb_dropout, out_indices = out_indices)
        self.neck = nn.Sequential(
            nn.Conv2d(
                en_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[1],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[1]),
            nn.Conv2d(
                neck_dims[1],
                neck_dims[2],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[2]),
        )
        self.decoder = nn.ModuleList()
        for i in range(de_depth):
            trans = Transformer(dim = neck_dims[2], depth = 1, heads=de_heads, dim_head=de_dim_head, mlp_dim=de_mlp_dim, dropout=dropout, out_indices = -1)
            self.decoder.append(trans)
        self.Head = nn.Sequential(ConvTransHead(), nn.Sigmoid())
        # self.last_layer = nn.Sequential(Mlp(in_features=18, hidden_features=8, out_features=1), nn.Sigmoid())
        
    def forward(self, x):
        out = self.encoder(x)
        x = out[-1]
        # x = rearrange(x, 'b n d -> b d h w', h = self.patch_h_num)
        x = self.neck(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for ids, dec in enumerate(self.decoder):
            if ids == len(self.decoder) - 1:
                out0 = rearrange(out[0], 'b c h w -> b (h w) c')
                x = (x + out0)/2
            x = dec(x)[0]
        x = rearrange(x, 'b (h w) d -> b d h w', h = self.patch_h_num)
        result = self.Head(x)
        # x = rearrange(x, 'b (h w) (bh bw) -> b (h bh) (w bw)', h = self.patch_h_num, bh = self.patch_size)
        # x = torch.unsqueeze(x, dim = 1)
        # x = rearrange(x, 'b c h w -> b h w c')
        # out = self.last_layer(x)
        # out = rearrange(x, 'b h w c -> b c h w')
        return result


# if __name__ == "__main__":
#     # VIT-Large  设置了16个patch
#     ViTNet1 = ViT(image_size=512, patch_size=16, dim=1024, depth = 1, heads = 16, mlp_dim = 2048, out_indices = (0,)).cpu()
#     ViTNet2 = ViT(image_size=256, patch_size=16, dim=1024, depth = 1, heads = 16, mlp_dim = 2048, out_indices = (0,)).cpu()
#     img1 = torch.randn(1, 3, 512, 512).cpu()
#     img2 = torch.randn(1, 3, 256, 256).cpu()
#     embedings1 = ViTNet1(img1, reshape_out = False)[0]
#     embedings2 = ViTNet2(img2, reshape_out = False)[0]
#     CrossAtt = CrossAttention(dim = 1024, heads = 8, dim_head = 64)
#     output = CrossAtt(embedings1, embedings2)
#     print(output.size)
 
# if __name__ == '__main__':
#     vit_unet = VitUnet(image_size=512, patch_size=16, dim=1024, depth = 4, heads = 16, mlp_dim = 2048).cpu()
#     img = torch.randn(1, 3, 512, 512).cpu()
#     output = vit_unet(img)
#     print(output.size)