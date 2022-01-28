import torch
import torchvision
import cv2


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"
    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2,dim=(1,2,3))
        batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse,))
        return batch_psnr.mean()

# image_org = cv2.cvtColor(cv2.imread('/media/hrishi/data/WORK/FYND/super_resolution/code/BlindSR/0853.png'),cv2.COLOR_BGR2RGB)
# image_dup = cv2.imread('/media/hrishi/data/WORK/FYND/super_resolution/code/BlindSR/0853.png')

# image_org = torch.tensor(image_org,dtype=torch.float32).permute(2,0,1)[None,...].tile((10,1,1,1))/255.0
# image_dup = torch.tensor(image_dup,dtype=torch.float32).permute(2,0,1)[None,...].tile((10,1,1,1))/255.0


# calc_psnr = PSNR()
# print(calc_psnr(image_org.to('cuda'),image_dup.to('cuda')).item())

