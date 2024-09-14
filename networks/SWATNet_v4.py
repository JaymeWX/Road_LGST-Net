import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.common import nonlinearity, DecoderBlock
from networks.SWATNet_old import Block, PatchEmbed
from einops import rearrange


# class DinkNet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3, is_swat = True):
#         super(DinkNet34, self).__init__()

#         img_size, patch_size, in_chans, patch_embed_dim = 512, 16, 3, 256
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)

#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4

#         # self.dblock = Dblock(512)

#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])

#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 16, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(16, num_classes, 3, padding=1)

#         self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)

#         # self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_e2 = nn.ModuleList()
#         for i in range(4):
#             vit_block = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#             self.vit_block_e2.append(vit_block)

#         # self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#         self.vit_block_e3 = nn.ModuleList()
#         for i in range(6):
#             vit_block = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#             self.vit_block_e3.append(vit_block)

#         self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#         self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)

#     def vit_ops(self, x, vit_block):
#         x = rearrange(x, 'B C H W -> B H W C')
#         if isinstance(vit_block, nn.ModuleList):
#             for block in vit_block:
#                 x = block(x)
#         else:
#             x = vit_block(x)
#         x = rearrange(x, 'B H W C -> B C H W')
#         return x

#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)

#         e1 = self.encoder1(x)
#         e1 = self.vit_ops(e1, self.vit_block_e1)

#         e2 = self.encoder2(e1)
#         e2 = self.vit_ops(e2, self.vit_block_e2)

#         e3 = self.encoder3(e2)
#         e3 = self.vit_ops(e3, self.vit_block_e3)

#         e4 = self.encoder4(e3)

#         # Center
#         # e4 = self.dblock(e4)

#         # Decoder
#         d4 = self.decoder4(e4)
#         d4 = self.vit_ops(d4, self.vit_block_d4)

#         d3 = self.decoder3(d4) 
#         d3 = self.vit_ops(d3, self.vit_block_d3)

#         d2 = self.decoder2(d3) 
#         d2 = self.vit_ops(d2, self.vit_block_d2)

#         d1 = self.decoder1(d2)

#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)

#         return F.sigmoid(out)


# class DinkNet34(nn.Module):
#     def __init__(self, num_classes=1, is_swat = True):
#         super(DinkNet34, self).__init__()

#         img_size, patch_size, in_chans, patch_embed_dim = 512, 4, 3, 256
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)

#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4

#         # self.dblock = Dblock(512)

#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])

#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 16, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(16, num_classes, 3, padding=1)

#         self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)

#         # self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_e2 = nn.ModuleList()
#         for i in range(4):
#             vit_block = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#             self.vit_block_e2.append(vit_block)

#         # self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#         self.vit_block_e3 = nn.ModuleList()
#         for i in range(6):
#             vit_block = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#             self.vit_block_e3.append(vit_block)

#         self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#         self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)

#     def vit_ops(self, x, vit_block):
#         x = rearrange(x, 'B C H W -> B H W C')
#         if isinstance(vit_block, nn.ModuleList):
#             for block in vit_block:
#                 x = block(x)
#         else:
#             x = vit_block(x)
#         x = rearrange(x, 'B H W C -> B C H W')
#         return x

#     def forward(self, x):
 
#         e3 = self.patch_embed(x)
#         e3 = self.vit_ops(e3, self.vit_block_e3)

#         e4 = self.encoder4(e3)

#         # Center
#         # e4 = self.dblock(e4)

#         # Decoder
#         d4 = self.decoder4(e4)
#         d4 = self.vit_ops(d4, self.vit_block_d4)

#         d3 = self.decoder3(d4) 
#         d3 = self.vit_ops(d3, self.vit_block_d3)

#         d2 = self.decoder2(d3) 
#         d2 = self.vit_ops(d2, self.vit_block_d2)

#         d1 = self.decoder1(d2)

#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)

#         return F.sigmoid(out)


#单层 vit
# class DinkNet34(nn.Module):
#     def __init__(self, num_classes=1, is_swat = True):
#         super(DinkNet34, self).__init__()

#         filters = [64, 128, 256, 512]
#         # resnet = models.resnet34(pretrained=True)
#         # self.firstconv = resnet.conv1
#         # self.firstbn = resnet.bn1
#         # self.firstrelu = resnet.relu
#         # self.firstmaxpool = resnet.maxpool
#         # self.encoder1 = resnet.layer1
        
        
#         # self.encoder2 = resnet.layer2
#         # self.encoder3 = resnet.layer3
#         # self.encoder4 = resnet.layer4
                
#         img_size, patch_size, in_chans, patch_embed_dim = 512, 4, 3, 64
#         self.encoder1 = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
#         self.encoder2 = nn.Conv2d(filters[0], filters[1], kernel_size = (3, 3), stride = 2, padding = 1)
#         self.encoder3 = nn.Conv2d(filters[1], filters[2], kernel_size = (3, 3), stride = 2, padding = 1)
#         self.encoder4 = nn.Conv2d(filters[2], filters[3], kernel_size = (3, 3), stride = 2, padding = 1)
        

#         # self.dblock = Dblock(512)

#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])

#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 16, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(16, num_classes, 3, padding=1)

#         self.vit_block_e1 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)
#         self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)

#         self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
#         self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
#         self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)



#     def vit_ops(self, x, vit_block):
#         x = rearrange(x, 'B C H W -> B H W C')
#         if isinstance(vit_block, nn.ModuleList):
#             for block in vit_block:
#                 x = block(x)
#         else:
#             x = vit_block(x)
#         x = rearrange(x, 'B H W C -> B C H W')
#         return x

#     def forward(self, x):
#         # Encoder
#         # x = self.firstconv(x)
#         # x = self.firstbn(x)
#         # x = self.firstrelu(x)
#         # x = self.firstmaxpool(x)

#         e1 = self.encoder1(x)
#         e1 = self.vit_ops(e1, self.vit_block_e1)
#         # e1 = rearrange(e1, 'B C H W -> B H W C')
#         # e1 = self.vit_block1(e1)
#         # e1 = rearrange(e1, 'B H W C -> B C H W')

#         e2 = self.encoder2(e1)
#         e2 = self.vit_ops(e2, self.vit_block_e2)
#         # e2 = rearrange(e2, 'B C H W -> B H W C')
#         # e2 = self.vit_block2(e2)
#         # e2 = rearrange(e2, 'B H W C -> B C H W')

#         e3 = self.encoder3(e2)
#         e3 = self.vit_ops(e3, self.vit_block_e3)
#         # e3 = rearrange(e3, 'B C H W -> B H W C')
#         # e3 = self.vit_block3(e3)
#         # e3 = rearrange(e3, 'B H W C -> B C H W')

#         e4 = self.encoder4(e3)

#         # Center
#         # e4 = self.dblock(e4)

#         # Decoder
#         d4 = self.decoder4(e4)
#         d4 = self.vit_ops(d4, self.vit_block_d4)

#         d3 = self.decoder3(d4) 
#         d3 = self.vit_ops(d3, self.vit_block_d3)

#         d2 = self.decoder2(d3) 
#         d2 = self.vit_ops(d2, self.vit_block_d2)

#         d1 = self.decoder1(d2)

#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)

#         return F.sigmoid(out)




class SWATNet(nn.Module):
    def __init__(self, num_classes=1, is_swat = True):
        super(SWATNet, self).__init__()

        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        
        
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
                
        img_size, patch_size, in_chans, patch_embed_dim = 512, 4, 3, 64
        self.encoder1 = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        self.encoder2 = nn.Conv2d(filters[0], filters[1], kernel_size = (3, 3), stride = 2, padding = 1)
        self.encoder3 = nn.Conv2d(filters[1], filters[2], kernel_size = (3, 3), stride = 2, padding = 1)
        self.encoder4 = nn.Conv2d(filters[2], filters[3], kernel_size = (3, 3), stride = 2, padding = 1)
        

        # self.dblock = Dblock(512)

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
        # self.vit_block_e2 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
        # self.vit_block_e3 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)


        self.vit_block_e2 = nn.ModuleList()
        for i in range(4):
            vit_block = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
            self.vit_block_e2.append(vit_block)

        self.vit_block_e3 = nn.ModuleList()
        for i in range(6):
            vit_block = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
            self.vit_block_e3.append(vit_block)

        self.vit_block_d4 = Block(256, 8, patch_w_num = 32, is_SWAT = is_swat)
        self.vit_block_d3 = Block(128, 4, patch_w_num = 64, is_SWAT = is_swat)
        self.vit_block_d2 = Block(64, 2, patch_w_num = 128, is_SWAT = is_swat)



    def vit_ops(self, x, vit_block):
        x = rearrange(x, 'B C H W -> B H W C')
        if isinstance(vit_block, nn.ModuleList):
            for block in vit_block:
                x = block(x)
        else:
            x = vit_block(x)
        x = rearrange(x, 'B H W C -> B C H W')
        return x

    def forward(self, x):
        # Encoder
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e1 = self.vit_ops(e1, self.vit_block_e1)
        # e1 = rearrange(e1, 'B C H W -> B H W C')
        # e1 = self.vit_block1(e1)
        # e1 = rearrange(e1, 'B H W C -> B C H W')

        e2 = self.encoder2(e1)
        e2 = self.vit_ops(e2, self.vit_block_e2)
        # e2 = rearrange(e2, 'B C H W -> B H W C')
        # e2 = self.vit_block2(e2)
        # e2 = rearrange(e2, 'B H W C -> B C H W')

        e3 = self.encoder3(e2)
        e3 = self.vit_ops(e3, self.vit_block_e3)
        # e3 = rearrange(e3, 'B C H W -> B H W C')
        # e3 = self.vit_block3(e3)
        # e3 = rearrange(e3, 'B H W C -> B C H W')

        e4 = self.encoder4(e3)

        # Center
        # e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4)
        d4 = self.vit_ops(d4, self.vit_block_d4)

        d3 = self.decoder3(d4) 
        d3 = self.vit_ops(d3, self.vit_block_d3)

        d2 = self.decoder2(d3) 
        d2 = self.vit_ops(d2, self.vit_block_d2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
