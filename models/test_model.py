from .base_model import BaseModel
from . import networks
import torch
from torch.autograd import Variable
from util import util
from util.metrics import calculate_psnr
from skimage.metrics import structural_similarity as SSIM

class TestModel(BaseModel):
    def name(self):
        return 'TrainModel'

    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        self.input_blur = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        which_epoch = opt.which_epoch
        self.model_names = ['G']
        # self.load_networks(which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_blur = input['blurred']
        self.input_blur.resize_(input_blur.size()).copy_(input_blur)
        self.image_paths = input['blur_path']
        self.input_sharp = input['sharp']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_blur)
            self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_images_and_metrics(self):
        fake = util.tensor2im(self.fake_B.data)[:720, :, :]
        real = util.tensor2im(self.input_sharp)
        psnr = calculate_psnr(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        return fake, psnr, ssim