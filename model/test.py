import torch,torchvision
import cv2
import matplotlib.pyplot as plt

org_img = cv2.imread('/media/hrishi/data/WORK/FYND/super_resolution/dataset/test/ecom_ds_small/val/hr/184.jpg')
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
h,w = org_img.shape[:2]
print(h,w)
org_img = torch.from_numpy(org_img)
# plt.imshow(org_img.numpy())
# plt.show()
# plt.close()
crop_img = torchvision.transforms.CenterCrop((h//2,w//2))(org_img.permute(2,0,1))
cv2.imwrite('input.png',cv2.cvtColor(crop_img.permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR))
# plt.imshow(crop_img.permute(1,2,0).numpy())
# plt.show()
# plt.close()


ffted_img = torch.fft.fft2(crop_img,(h//2,w//2),(1,2),'ortho')
ffted_img = torch.fft.fftshift(ffted_img)
# ffted_img = torch.fft.fftn(org_img,(h,w),(1,2),'ortho')
ffted_img_real,ffted_img_imag = ffted_img.real,ffted_img.imag
# upsampler = torchvision.transforms.Resize((h,w),interpolation=torchvision.transforms.InterpolationMode.NEAREST)
padding_1 = torchvision.transforms.Pad((w//4,h//4,w//4,h//4),padding_mode='reflect')
padding_2 = torchvision.transforms.Pad((w//4,h//4,w//4,h//4),padding_mode='reflect')
# ffted_img_real_up,ffted_img_imag_up = upsampler(ffted_img_real),upsampler(ffted_img_imag)
# print(ffted_img_real.max(),ffted_img_imag.max())
ffted_img_real_up,ffted_img_imag_up = padding_1(ffted_img_real),padding_2(ffted_img_imag)
ffted_img_up = torch.complex(ffted_img_real_up,ffted_img_imag_up)
print(ffted_img_up.shape)
# ffted_img_up = torch.complex(ffted_img_real,ffted_img_imag)
pred = torch.fft.ifft2(torch.fft.ifftshift(ffted_img_up),s = (h,w),dim=(1,2),norm='ortho')
cv2.imwrite('result.png',cv2.cvtColor(pred.real.permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR))
# print(pred.shape)
# plt.imshow(pred.abs().permute(1,2,0).numpy())
# plt.savefig('result.png')
# plt.close()
# .permute(1,2,0)


# inp = torch.randn((1,3,256,256))
# out = SR(3,3)(inp)
# summ = get_model_summary(SR(3,3),(3,256,256))
# print(summ)

# s=(in_h*self.scale,in_w*self.scale)
# import torchvision
# import cv2
# import numpy as np
# image = torchvision.io.read_image('/media/hrishi/data/WORK/FYND/super_resolution/dataset/test/input/180.jpg').type(torch.float32)/255.
# out = ModifiedFFTUpsampler(3,3,64).to('cuda')(image[None,...].to('cuda'))
# print(out.shape)
# # print(torch.mean(out[2]))
# test_inp = nn.Upsample(scale_factor=2)(image[None,...])
# diff = (test_inp[0,:,:,:]).permute(1,2,0).numpy()-(out[0,:,:,:]).permute(1,2,0).numpy()
# print(diff[...,0])
# cv2.imwrite('diff.png',np.abs(diff[...,0])*255)
# out = cv2.imwrite('test.png',cv2.cvtColor((out[0,:,:,:]*255).type(torch.uint8).permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR))