import argparse
import os
import torch

class BaseOptions():
    """
    This class defines options used during both training and test time.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        self.parser.add_argument('--data_root', type=str, default="/home/limingzhu/DVD_Dataset/IMG_0039/input", help='path to selected sharp frames')
        self.parser.add_argument('--fineSize', type=int, default=256, help='crop image to this size')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--which_model_netD', type=str, default='n_layers', help='selects model to use for netD. [basic|n_layers]')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG[unet_256|unet_128]')
        self.parser.add_argument('--n_layers_D', type=int, default=5, help='use 5 if which_model_netD==n_layers, 3 if which_model_netD==basic')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--gan_type', type=str, default='wgan-gp', help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--name', type=str, default='VideoDeblur',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='test',
                                 help='chooses how datasets are loaded. [train | test]')
        self.parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints/", help='models are saved here')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--model', type=str, default='test', help='chooses which model to use. pix2pix, test, content_gan')

        self.parser.add_argument('--loadSizeX', type=int, default=640, help='scale images to this size')
        self.parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
        self.parser.add_argument('--learn_residual', action='store_true',
                                 help='if specified, model would learn only the residual to the input')

        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8013, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')

        self.initialized = True