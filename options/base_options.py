import argparse
import os
import torch
from util import util

class BaseOptions():
    """
    This class defines options used during both training and test time.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic parameters
        self.parser.add_argument('--data_root', type=str, default="E:/GoPro_Large/sharp/GOPR0384_11_05", help='path to save selected sharp frames')
        self.parser.add_argument('--video_root', type=str, default="E:/GoPro_Large/test/blur/GOPR0384_11_05", help='path to blurry video sequences')
        self.parser.add_argument('--sharp_root', type=str, default="E:/GoPro_Large/test/sharp/GOPR0384_11_05", help='path to the sharp video sequences')
        self.parser.add_argument('--fineSize', type=int, default=1280, help='crop image to this size')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--which_model_netD', type=str, default='n_layers', help='selects model to use for netD. [basic|n_layers]')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG[unet_256|unet_128]')
        self.parser.add_argument('--n_layers_D', type=int, default=5, help='use 5 if which_model_netD==n_layers, 3 if which_model_netD==basic')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--gan_type', type=str, default='wgan-gp', help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--name', type=str, default='VideoDeblur',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='test',
                                 help='chooses how datasets are loaded. [train | test]')
        self.parser.add_argument('--use_validation', action='store_true', help='if true, use validation')
        self.parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints/", help='models are saved here')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--model', type=str, default='test', help='chooses which model to use. content_gan, test')
        self.parser.add_argument('--kernel_size', type=int, default=21, help='the size of kernel [21 | 31 | 41]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
