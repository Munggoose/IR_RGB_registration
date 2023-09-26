import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np

def L1_smooth_loss(x, y):
    abs_diff = torch.abs(x - y)
    abs_diff_lt_1 = torch.le(abs_diff, 1)
    return torch.mean(torch.where(abs_diff_lt_1, 0.5 * abs_diff ** 2, abs_diff - 0.5))



def SSIM_loss(x, y, size=1):
    # C = (K*L)^2 with K = max of intensity range (i.e. 255). L is very small
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, size, 1, padding=0)
    mu_y = F.avg_pool2d(y, size, 1, padding=0)

    sigma_x = F.avg_pool2d(x ** 2, size, 1, padding=0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, size, 1, padding=0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, size, 1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1).mean()


def smoothness_loss(deformation, img=None, alpha=0.0):
    """
    Calculate the smoothness loss of the given defromation field
    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    
    """
    
    diff_1 = torch.abs(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    # diff_2 = torch.abs((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_2 = torch.abs(deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1])
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    
    
    weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss

def MMIR_triplet_loss(warp_fe,img_a_fe, img_b_fe):
    # big = F.l1_loss(warp_fe,img_a_fe)
    # small = F.l1_loss(warp_fe, img_b_fe)
    big = F.mse_loss(warp_fe,img_a_fe)
    small = F.mse_loss(warp_fe, img_b_fe)
    return 1 - (big - small)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss