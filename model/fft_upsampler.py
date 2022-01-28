from model.dcn import DeformableConv2d
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



        

            

