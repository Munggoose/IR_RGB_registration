import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .netT import UnetGenerator as MT
from .netR.affine_stn import AffineSTN
from .netT.utils import GANLoss
import itertools


class AffineRegistrator(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.device='cuda'
        self.cfg = cfg
        self.netT = MT.UnetGenerator(1,1,8,64)
        self.netR = AffineSTN(cfg)
        self.netD = MT.NLayerDiscriminator(2, 64, n_layers=3)
        
        self.criterionGAN = GANLoss('vanilla').to(self.device)  # define GAN loss.
        self.criterionL1 = torch.nn.L1Loss()
        self.optimizers = []
        self.setup_optimizers()
        
    def reset_weights(self):
        # We have tested what happens if we reset the discriminator/translation network's weights during training.
        # This eventually will results in th
        self.init_weights(self.netT,'normal', 0.02)
        self.init_weights(self.netD, 'normal', 0.02)
    
    
    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
    

    def setup_optimizers(self):
        opt = self.cfg

        # Define optimizer for the registration network:
        self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()),
                                            lr=opt.lr, betas=(0.5, 0.999), )
        # Define optimizer for the translation network:
        self.optimizer_T = torch.optim.Adam([{'params': self.netT.parameters(), 'betas': (0.5, 0.999),
                                            'lr': opt.lr}])
        # Define optimizer for the discriminator network:
        d_params = self.netD.parameters()

        self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr, betas=(0.5, 0.999))

        self.optimizers.append(self.optimizer_T)
        self.optimizers.append(self.optimizer_D)
        self.optimizers.append(self.optimizer_R)
    
    
    def forward(self,ir,rgb):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ir = ir
        self.rgb = rgb
        
        self.fake_rgb = self.netT(self.ir) ## ir -> rgb 나중에 rgb->ir 로 변경해보자

        warped_ir,ir_grid, reg_term = self.netR(self.ir, self.rgb)
        self.warped_ir = warped_ir
        self.ir_grid = ir_grid
        self.stn_reg_term = reg_term
        self.fake_TR_rgb  = self.netR.warping(self.fake_rgb,ir_grid)

        # Registration first -- Then --> Translation
        self.fake_RT_rgb = self.netT(self.warped_ir)
        # Translation first  -- Then --> Registration

        return warped_ir, reg_term
        # if self.tb_visualizer:
        #     with torch.no_grad():
        #         self.deformation_field_A_to_B = self.netR.module.get_grid(self.real_A, self.real_B)
    
    
    def backward_T_and_R(self):
        """Calculate GAN and L1 loss for the translation and registration networks."""
        # Registration first (TR):
        # ----> Reconstruction loss:
        self.loss_L1_TR = self.cfg.lambda_recon * self.criterionL1(self.fake_TR_rgb, self.rgb)
        # ----> GAN loss:
        fake_AB_t = torch.cat((self.ir, self.fake_TR_rgb), 1)
        pred_fake = self.netD(fake_AB_t)
        self.loss_GAN_TR = self.cfg.lambda_GAN * self.criterionGAN(pred_fake, True)
        # --------> Multi-scale discrimnaotr

        # Translation First:
        # ----> Reconstruction loss:
        self.loss_L1_RT = self.cfg.lambda_recon * self.criterionL1(self.fake_RT_rgb, self.rgb)
        
        # ----> GAN loss:
        fake_AB_t = torch.cat((self.ir, self.fake_RT_rgb), 1)
        pred_fake = self.netD(fake_AB_t)
        self.loss_GAN_RT = self.cfg.lambda_GAN * self.criterionGAN(pred_fake, True)
        # --------> Multi-scale discrimnaotr

        self.loss_smoothness = self.cfg.lambda_smooth * self.stn_reg_term

        loss = self.loss_L1_TR + self.loss_L1_RT  + self.loss_GAN_TR + self.loss_GAN_RT + self.loss_smoothness
        loss.backward()

        return (self.loss_GAN_TR + self.loss_GAN_RT), (self.loss_L1_RT + self.loss_L1_TR), (self.loss_L1_RT + self.loss_L1_TR)
    
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        real_AB = torch.cat((self.ir, self.rgb), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # --------> Multi-scale discrimnaotr

        # Registration Firsts (TR):
        # ----> Fake
        fake_AB = torch.cat((self.ir, self.fake_TR_rgb), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake_TR = self.criterionGAN(pred_fake, False)
        # --------> Multi-scale discrimnaotr


        # Translation First (RT):
        # ----> Fake
        fake_AB = torch.cat((self.ir, self.fake_RT_rgb), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake_RT = self.criterionGAN(pred_fake, False)
        # --------> Multi-scale discrimnaotr


        # combine loss and calculate gradients
        self.loss_D = 0.5 * self.cfg.lambda_GAN * (loss_D_real + self.loss_D_fake_TR + self.loss_D_fake_RT)
        self.loss_D.backward()

        return self.loss_D
    
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # self.forward()  # TR(I_a) and RT(I_a)
        # Backward D
        self.set_requires_grad([self.netT, self.netR], False)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        D_loss = self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.set_requires_grad([self.netT, self.netR], True)

        # Backward translation and registration networks
        self.set_requires_grad([self.netD], False)
        self.optimizer_R.zero_grad()
        self.optimizer_T.zero_grad()  # set G_A and G_B's gradients to zero
        
        gan_loss, reg_ir_loss , reg_rgb_loss = self.backward_T_and_R()  # calculate gradients for translation and registration networks
        self.optimizer_R.step()
        self.optimizer_T.step()
        self.set_requires_grad([self.netD], True)

        # Update tb visualizer on each iteration step - if enabled
        # if self.tb_visualizer is not None:
        #     self.tb_visualizer.iteration_step()

        
        return D_loss, gan_loss, reg_ir_loss, reg_rgb_loss