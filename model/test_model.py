import torch
import torch.nn as nn
from model.fft_upsampler import ModifiedFFTUpsampler
from run_utils import get_model_summary
class SR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(SR, self).__init__()
        self.in_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.freq_upsampler = ModifiedFFTUpsampler(ngf,ngf,ngf)
        # self.space_upsampler = nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(ngf, ngf, kernel_size=3, padding=1))
        # self.out = nn.Sequential(nn.ReflectionPad2d(3),
        #                            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
    def forward(self,input):
        x_in = self.in_layer(input)
        x_freq = self.freq_upsampler(x_in)
        # x_spat = self.space_upsampler(x)
        result = self.out(x_freq)
        return result



