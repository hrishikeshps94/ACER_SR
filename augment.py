from globals import TRAINING_CROP_SIZE
import random


TRAINING_CROP_SIZE = 256

def random_crop_coordinate(image_shape):
    h,w,_ = image_shape
    hr_x0,hr_y0 = random.randint(0,w-TRAINING_CROP_SIZE),random.randint(0,h-TRAINING_CROP_SIZE)
    hr_x1,hr_y1 = hr_x0+TRAINING_CROP_SIZE,hr_y0+TRAINING_CROP_SIZE
    lr_x0,lr_y0 = hr_x0//2,hr_y0//2
    lr_x1,lr_y1 = lr_x0+TRAINING_CROP_SIZE//2,lr_y0+TRAINING_CROP_SIZE//2
    hr_coor = (hr_x0,hr_y0,hr_x1,hr_y1)
    lr_coor = (lr_x0,lr_y0,lr_x1,lr_y1)
    return hr_coor,lr_coor

def random_crop(hr_img,lr_img):
    image_shape = hr_img.shape
    hr_coor,lr_coor = random_crop_coordinate(image_shape)
    hr_cropped = hr_img[hr_coor[1]:hr_coor[3],hr_coor[0]:hr_coor[2],:]
    lr_cropped = lr_img[lr_coor[1]:lr_coor[3],lr_coor[0]:lr_coor[2],:]
    return hr_cropped,lr_cropped