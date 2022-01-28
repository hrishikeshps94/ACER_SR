import os
import random
import torch
import torchvision
import random
import math,cv2
from torch.utils.data import Dataset
from torch.nn.functional import pad
import PIL
import numpy as np
from torchvision import transforms
from globals import TRAINING_CROP_SIZE, SCALE_FACTOR, KERNEL_SIZE
from degradation import Degradation,random_add_jpg_compression

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f'Device used = {device}')

class TrainDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)
        self.transforms_list = [transforms.InterpolationMode.BILINEAR,transforms.InterpolationMode.BICUBIC]
        self.alias_cond = [True,False]
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomCrop(TRAINING_CROP_SIZE*1.5, pad_if_needed=True)
        ])
        self.hr_center_crop = torchvision.transforms.CenterCrop(TRAINING_CROP_SIZE)
        self.lr_center_crop = torchvision.transforms.CenterCrop(TRAINING_CROP_SIZE//2)
        self.tensor_convert = torchvision.transforms.ToTensor()
        self.image_convert = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file).convert('RGB')
        h,w = img.size
        # if h%4==0 and w%4==0 and h//4>=int(TRAINING_CROP_SIZE*1.5) and w//4>=int(TRAINING_CROP_SIZE*1.5):
        #     print('condtion statisfied')
        image_tuple = torchvision.transforms.functional.five_crop(img,(h//2,w//2))
        img = image_tuple[random.randint(0,4)]
        img = self.image_transform(img)
        hr_img = self.hr_center_crop(img)
        hr_img = self.tensor_convert(hr_img)
        # h,w = hr_img.shape[1:3]
        lr_img = torchvision.transforms.Resize(int(TRAINING_CROP_SIZE*1.5)//2,self.transforms_list[random.randint(0,1)],antialias=True)(img)
        # self.alias_cond[random.randint(0,1)]
        # lr_img = torchvision.transforms.Resize((h//2,w//2),self.transforms_list[random.randint(0,1)],antialias=True)(img)
        lr_img = cv2.cvtColor(np.asarray(lr_img),cv2.COLOR_RGB2BGR)/255.
        lr_img = cv2.cvtColor(random_add_jpg_compression(lr_img),cv2.COLOR_BGR2RGB)
        lr_img = self.tensor_convert(lr_img)
        lr_img = self.lr_center_crop(lr_img)
        return {'lowres_img': lr_img,
        'ground_truth_img': hr_img
        }



class ValidDataset(Dataset):
    def __init__(self, folder):
        self.lr_image_files = []
        self.hr_image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in sorted(file_names):
              if dir_path.endswith('lr'):
                file_name = os.path.join(dir_path, f)
                self.lr_image_files.append(file_name)
              elif dir_path.endswith('hr'):
                file_name = os.path.join(dir_path, f)
                self.hr_image_files.append(file_name)
        self.tensor_convert = torchvision.transforms.ToTensor()
        self.image_convert = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.lr_image_files)

    def __getitem__(self, item):
        lr_image_file = self.lr_image_files[item]
        hr_image_file = self.hr_image_files[item]
        lr_img = PIL.Image.open(lr_image_file).convert('RGB')
        hr_img = PIL.Image.open(hr_image_file).convert('RGB')
        lr_img = self.tensor_convert(lr_img)
        hr_img = self.tensor_convert(hr_img)

        return {
            'lowres_img': lr_img,
            'ground_truth_img': hr_img
            # 'img_size': [new_h,new_w],
        }

class TestDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                print(f)
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)
        self.tensor_convert = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file).convert('RGB')

        # If height > width, we flip the image
        flipped = False
        resize = False
        if img.height > img.width:
            img = img.transpose(PIL.Image.TRANSPOSE)
            flipped = True
        org_h,org_w = img.size

        img = self.tensor_convert(img)
        return {'lowres_img': img,
            'img_name': os.path.basename(image_file),
            'flipped': flipped,
            'resize': resize,
            'org_shape':(org_h,org_w)
        }


