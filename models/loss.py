import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

class DiscLossWGANGP():
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self, opt, tensor):
        super(DiscLossWGANGP, self).__init__(opt, tensor)
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty

def init_loss(opt, tensor):
    """
    initialize loss, in this task, we use perceptual loss for netG, and WGAN-GP loss for netD
    """
    if opt.model == 'content_gan':
        content_loss = PerceptualLoss(nn.MSELoss())
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.gan_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP(opt, tensor)
    else:
        raise ValueError("GAN [%s] not recognized." % opt.gan_type)
    return disc_loss, content_loss