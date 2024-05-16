import torch
from torch import nn
 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
 



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
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
 
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
 
        attn = self.attend(dots)
        attn = self.dropout(attn)
 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., out_indices = (9, 14, 19, 23)):
        super().__init__()
        self.out_indices = out_indices
        assert self.out_indices[-1] == depth - 1
 
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
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
 
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
 
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
 
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, out_indices=out_indices)
 
        self.out = Rearrange("b (h w) c->b c h w", h=image_height//patch_height, w=image_width//patch_width)
 
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
 
        out = self.transformer(x)
 
        for index, transformer_out in enumerate(out):
            # delete cls_tokens and transform output to [b, c, h, w]
            out[index] = self.out(transformer_out[:,1:,:])
 
        return out
 
 
class PUPHead(nn.Module):
    def __init__(self, num_classes):
        super(PUPHead, self).__init__()
        
        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_3= nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )        
        self.UP_stage_4= nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_5= nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
    
        self.cls_seg = nn.Conv2d(32, num_classes, 3, padding=1)
 
    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        x = self.UP_stage_4(x)
        x = self.UP_stage_5(x)
        x = self.cls_seg(x)
        return x
 
class SETR(nn.Module):
    def __init__(self, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., out_indices = (9, 14, 19, 23)):
        super(SETR, self).__init__()
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.VIT = ViT( image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, 
                        channels = channels, dim_head = dim_head, dropout = dropout, emb_dropout = emb_dropout, out_indices = out_indices)
 
        
        self.Head = nn.ModuleDict()
 
        for index, indices in enumerate(self.out_indices):
            self.Head["Head"+str(indices)] = PUPHead(num_classes)

        # self.last_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=2, stride=2, ), nn.Sigmoid())
        self.last_layer = nn.Sequential(Mlp(in_features=4, out_features=1), nn.Sigmoid())

    def forward(self, x):
        VIT_OUT = self.VIT(x)
 
        out = []
        for index, indices in enumerate(self.out_indices):
            # 最后一个是最后层的输出
            out.append(self.Head["Head"+str(indices)](VIT_OUT[index]))

        out = torch.cat(out, dim = 1)
        out = out.permute(0, 2, 3, 1)
        out = self.last_layer(out)
        out = out.permute(0, 3, 1, 2)
        return out
 
 
# if __name__ == "__main__":
#     # VIT-Large  设置了16个patch
#     SETRNet = SETR(num_classes=3, image_size=256, patch_size=256//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23)).cpu()
#     img = torch.randn(1, 3, 256, 256).cpu()
#     preds = SETRNet(img)
#     for output in preds:
#         print("output: ",output.size())
 