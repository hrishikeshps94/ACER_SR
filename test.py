import os,torch,math
from degradation import Degradation
import torchvision
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f'Device used= {device}')
from globals import *
from dataset import TestDataset
from model.acernet import SR
from torch.utils.data import DataLoader
import imageio
import tqdm


class Test():
    def __init__(self, args):
        self.args = args
        self.init_model_for_testing()
        self.restore_models_for_testing()
        self.init_test_data()
        self.launch_test()
    def init_model_for_testing(self):
        self.generator = SR(3,3)
        self.generator = self.generator.to(device)

    def init_test_data(self):
        test_folder = self.args.input
        print(os.listdir(test_folder))
        test_dataset = TestDataset(test_folder)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=os.cpu_count())

    def restore_models_for_testing(self):
        checkpoint_folder = self.args.checkpoint_folder
        generator_checkpoint_filename = os.path.join(checkpoint_folder, 'generator_best.pth')
        if (not os.path.exists(generator_checkpoint_filename)):
            print("Error: could not locate network checkpoints. Make sure the files are in the right location.")
            print(f"The generator checkpoint should be at {generator_checkpoint_filename}")
            exit()
        data = torch.load(generator_checkpoint_filename)
        self.generator.load_state_dict(data['generator_state_dict'])

    def launch_test(self):
        for batch in tqdm.tqdm(self.test_dataloader):
            lowres_img = batch['lowres_img'].to(device)
            image_name = batch['img_name']
            flipped = batch['flipped']
            resized = batch['resize']
            org_h,org_w = batch['org_shape']
            batch_size, channels, img_height, img_width = lowres_img.size()
            print(lowres_img.shape)

            with torch.no_grad():
                highres_output = self.generator(lowres_img)
                if flipped:
                    highres_output = highres_output.permute(0, 1, 3, 2)
                highres_image = (highres_output[0]*255.0).permute(1, 2, 0).clamp(0.0, 255.0).type(torch.uint8).cpu().numpy()
                output_folder = self.args.output
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_image_name = str.split(image_name[0], '.')[0] + '.png'
                output_file = os.path.join(output_folder, output_image_name)
                imageio.imwrite(output_file, highres_image)
                print(f"Saving output image at {output_file}.")