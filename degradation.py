from numpy.core.fromnumeric import shape
import torch
import cv2,random
import numpy as np
from torch.nn.functional import conv2d
from globals import SCALE_FACTOR, KERNEL_SIZE

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
class Degradation:
    def __init__(self, kernel_size=KERNEL_SIZE, theta=0.0, sigma=[1.0, 1.0]):
        self.kernel_size = kernel_size
        self.theta = torch.tensor([theta]).to(device)
        self.sigma = torch.tensor(sigma).to(device)

        self.kernel = None
        # self.build_kernel()


    def set_parameters(self, sigma, theta):
        self.sigma = sigma.to(device)#Check whether this works or not
        self.theta = theta.to(device)

    def build_kernel(self):
        kernel_radius = self.kernel_size // 2
        kernel_range = torch.linspace(-kernel_radius, kernel_radius, self.kernel_size).to(device)
        horizontal_range = kernel_range[None].repeat((self.kernel_size, 1))
        vertical_range = kernel_range[:, None].repeat((1, self.kernel_size))

        cos_theta = self.theta.cos()
        sin_theta = self.theta.sin()

        cos_theta_2 = cos_theta ** 2
        sin_theta_2 = sin_theta ** 2

        sigma_x_2 = 2.0 * (self.sigma[0] ** 2)
        sigma_y_2 = 2.0 * (self.sigma[1] ** 2)

        a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
        b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
        c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

        gaussian = lambda x,y: (- ( a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2))).exp()

        kernel = gaussian(horizontal_range, vertical_range)
        kernel = kernel / kernel.sum()
        self.kernel = kernel
        return self.kernel.to(device)

    def get_kernel(self):
        self.build_kernel()
        return self.kernel

    def get_features(self):
        self.build_kernel()
        return torch.reshape(self.kernel, (self.kernel_size ** 2,))

    def apply(self, img, scale=SCALE_FACTOR):
        weights = torch.zeros(3,3,self.kernel_size, self.kernel_size).to(device)
        self.build_kernel()

        for c in range(3):
            weights[c, c, :, :] = self.kernel
        conv_img = conv2d(img[None], weights,padding='same') # 3 filters for 3 channels as output
        scale_factor = int(scale)
        lr_img = conv_img[0, :, ::scale_factor, ::scale_factor]
        return lr_img

##### JPEG Compression ##############################

def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    encode_param = [[cv2.IMWRITE_JPEG_QUALITY, quality],[cv2.IMWRITE_WEBP_QUALITY, quality]]
    format = ['.jpg','.webp']
    idx = random.randint(0,1)
    _, encimg = cv2.imencode(format[idx], img * 255., encode_param[idx])
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img

def random_add_jpg_compression(img, quality_range=(60,100)):
    """Randomly add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)


