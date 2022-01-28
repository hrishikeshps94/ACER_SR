import os
import argparse
import torch
from generator_train import Train as gen_train
from test import Test
from globals import SCALE_FACTOR
from datetime import datetime
def main(args):
    if args.mode=='test':
        print("Test mode")
        Test(args)
    elif args.mode=='train':
        if args.network_type=='generator':
            gen_train(args)
        else:
            print("select a valid mode network type generator/discriminator")
    else:
        print("select a valid mode test/train")
    return None


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='test')

    main_path = os.path.dirname(os.path.abspath(__file__))
    # print(main_path)
    # ------------------------------------------------
    # Testing arguments
    # ------------------------------------------------

    # Input folder for test images
    # arg_parser.add_argument('--input', type=str, default=os.path.join('/media/hrishi/data/WORK/FYND/scrape/scraped/', 'dataset', 'val','lr'))
    # arg_parser.add_argument('--input', type=str, default=os.path.join('/media/hrishi/data/WORK/FYND/super_resolution/dataset/test/', 'ecom_ds_small', 'val','lr'))
    arg_parser.add_argument('--input', type=str, default='/media/hrishi/data/WORK/FYND/super_resolution/dataset/DIV2K_valid_LR_bicubic_X2/DIV2K_valid_LR_bicubic/X2/')
    # Output folder for test results
    arg_parser.add_argument('--output', type=str, default=os.path.join(main_path, 'output'))

    # Patch size for patch-based testing of large images.
    # Make sure the patch size is small enough that your GPU memory is sufficient.
    arg_parser.add_argument('--patch_size', type=int, default=128)
    current_time = datetime.now()
    # Checkpoint folder that contains the generator.pth and discriminator.pth checkpoint files.
    arg_parser.add_argument('--checkpoint_folder', type=str,
                            default=os.path.join(main_path, 'checkpoints', 'x' + str(SCALE_FACTOR) + '__sr__'+'acer_l1_fft'))

    # ------------------------------------------------
    # Training arguments
    # ------------------------------------------------

    # Log folder where Tensorboard logs are saved
    arg_parser.add_argument('--log_name', type=str,
                            default=str(SCALE_FACTOR) + '__sr__'+'acer_l1_fft')
    arg_parser.add_argument('--epochs', type=int,
                            default=100)

    # Folders for training and validation datasets.
    arg_parser.add_argument('--train_input', type=str, default=os.path.join('/media/hrishi/data/WORK/FYND/super_resolution', 'dataset', 'val'))
    arg_parser.add_argument('--valid_input', type=str, default=os.path.join('/media/hrishi/data/WORK/FYND/super_resolution/dataset/test/', 'ecom_ds_small', 'val'))
    # Define whether we use only the generator or the whole pipeline with the discriminator for training.
    arg_parser.add_argument('--network_type', type=str, choices=['generator', 'discriminator'], default='generator')

    arg_list = arg_parser.parse_args()
    main(arg_list)
