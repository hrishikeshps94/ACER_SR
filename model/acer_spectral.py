import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.dcn import DeformableConv2d
from model.upsampler import DWTUpsampler

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class SRBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(SRBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        # self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        # res = self.palayer(res)
        res += x
        return res


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class SR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(SR, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = SRBlock(default_conv, ngf * 4, 3)

        ###### upsample
        self.up1 = nn.Sequential(DWTUpsampler(ngf*4, ngf*2,nr_feat=ngf),nn.ReLU(True))
        self.up2 = nn.Sequential(DWTUpsampler(ngf*2, ngf,nr_feat=ngf), nn.ReLU(True))
        self.up3 = nn.Sequential(DWTUpsampler(ngf, 3,nr_feat=ngf))


        self.dcn_block = DeformableConv2d(256,256)


        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        self.mix3 = Mix(m=-0.2)

    def forward(self, input):

        # x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(input) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256]
        x_up2_mix = self.mix3(x_down1,x_up2)
        out = self.up3(x_up2_mix) # [bs,  3, 512, 512]

        return out