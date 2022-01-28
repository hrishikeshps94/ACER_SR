from pyexpat import model
from model.dcn import DeformableConv2d
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn as nn
import torch



class ModifiedFFTUpsampler(nn.Module):
    def __init__(self,in_channels,out_channels,nr_feat,scale=2):
        super(ModifiedFFTUpsampler,self).__init__()
        self.scale = scale
        self.inp_process = nn.Sequential(*[DeformableConv2d(in_channels,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,nr_feat,bias=True)])
        self.real_upsampler = nn.Sequential(*[DeformableConv2d(nr_feat,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,out_channels,bias=True)])
        self.imag_upsampler = nn.Sequential(*[DeformableConv2d(nr_feat,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,out_channels,bias=True)])
    def forward(self,input):
        in_h,in_w = input.shape[2:]
        inp_feat = self.inp_process(input)
        upsampled = nn.functional.interpolate(inp_feat,size=(in_h*self.scale,in_w*self.scale))
        ffted = torch.fft.rfftn(upsampled,dim=(2,3),norm='ortho')
        ffted = torch.fft.fftshift(ffted)
        real_upsampled = self.real_upsampler(ffted.real)
        imag_upsampled = self.imag_upsampler(ffted.imag)
        upsampled = torch.complex(real_upsampled,imag_upsampled)
        out = torch.fft.irfftn(torch.fft.ifftshift(upsampled),dim=(2,3),norm='ortho')
        return out

class DWTUpsampler(nn.Module):
    def __init__(self,in_channels,out_channels,nr_feat,scale=2):
        super(DWTUpsampler,self).__init__()
        self.scale = scale
        self.inp_process = nn.Sequential(*[DeformableConv2d(in_channels,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,nr_feat,bias=True)])
        self.low_upsampler = nn.Sequential(*[DeformableConv2d(nr_feat,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,out_channels,bias=True)])
        self.high_upsampler = nn.Sequential(*[DeformableConv2d(nr_feat*3,nr_feat,bias=True),\
            DeformableConv2d(nr_feat,out_channels*3,bias=True)])
        self.DWT = DWTForward(wave='db4')
        self.IDWT  = DWTInverse(wave='db4')
    def forward(self,input):
        inp_feat = self.inp_process(input)
        upsampled = nn.functional.interpolate(inp_feat,scale_factor=2)
        LL,HF = self.DWT(upsampled)
        HF = torch.tensor_split(*HF,3,dim=2)
        HF = torch.cat([HF[0].squeeze(dim=2),HF[1].squeeze(dim=2),HF[2].squeeze(dim=2)],dim=1)
        low_upsampled = self.low_upsampler(LL)
        high_upsampled = self.high_upsampler(HF)
        high_upsampled = torch.tensor_split(high_upsampled,3,dim=1)
        high_upsampled = torch.cat([high_upsampled[0].unsqueeze(dim=2),high_upsampled[1].unsqueeze(dim=2),high_upsampled[2].unsqueeze(dim=2)],dim=2)
        out = self.IDWT((low_upsampled,[high_upsampled]))
        return out


        

            

