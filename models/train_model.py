from .base_model import BaseModel
from . import networks
import torch
from torch.autograd import Variable
from .loss import init_loss
import numpy as np
from util.metrics import calculate_psnr
from skimage.metrics import structural_similarity as SSIM

class TrainModel(BaseModel):
    def name(self):
        return 'TrainModel'

    def __init__(self, opt):
        super(TrainModel, self).__init__(opt)
        # define tensors
        self.input_sharp = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_blur = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        # self.input_V = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.loss_names = ['G_GAN', 'G_Content', 'D']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define a discriminator
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.old_lr = opt.lr
            # define loss functions
            self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

        if self.isTrain and opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        input_sharp = input['sharp'].to(self.device)
        input_blur = input['blur'].to(self.device)
        self.input_sharp.resize_(input_sharp.size()).copy_(input_sharp)
        self.input_blur.resize_(input_blur.size()).copy_(input_blur)
        self.image_paths = input['sharp_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A = Variable(self.input_blur)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_sharp)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.fake_B, self.real_B)
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_Content

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for iter_d in range(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self) :
        inp = self.tensor2im(self.real_A.data)
        fake = self.tensor2im(self.fake_B.data)
        real = self.tensor2im(self.real_B.data)
        psnr = calculate_psnr(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img