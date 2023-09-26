from .block import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..netT.utils import smoothness_loss

class NetR(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.oh = cfg.img_size[1]
        self.ow = cfg.img_size[0]
        
        self.cfg = cfg
        self.identity_grid = self.get_identity_grid(self.ow,self.oh)
        self.stage_1 = MFlownet(5)
        # self.ir_stage()
        # self.rgb_stage()
    
    
    def get_identity_grid(self,ow,oh):
        """Returns a sampling-grid that represents the identity transformation."""
        
        x = torch.linspace(-1.0, 1.0, ow)
        y = torch.linspace(-1.0, 1.0, oh)
        xx, yy = torch.meshgrid([y, x],indexing='ij')
        xx = xx.unsqueeze(dim=0) # 1(c) x matrix
        yy = yy.unsqueeze(dim=0) # 1(c) x matrix
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0) #batch size
        return identity
    
    
    def transform_grid(self,img_a, deformation, return_offsets_only=False):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Get Deformation Field
        b_size = img_a.size(0)
        
        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(deformation, (self.oh, self.ow), mode='bilinear',
                                                    align_corners=False)
        else:
            deformation_upsampled = deformation
            
        if return_offsets_only:
            resampling_grid = deformation_upsampled.permute([0, 2, 3, 1])
        else:
            resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])
            
        return resampling_grid
    
    
    def warping(self, x, resampling_grid):
        warped_images = F.grid_sample(x, resampling_grid, mode='bilinear', padding_mode='zeros',
                                                align_corners=False)
        return warped_images
    
    
    def _calculate_regularization_term(self, deformation, img):
        """Calculate the regularization term of the predicted deformation.
        The regularization may-be applied to different resolution for larger images."""
        img = None if img is None else img.detach()

        reg = 0.0
        factor = 1.0
        deformation_resized = deformation
        img_resized = img
        reg += factor * smoothness_loss(deformation_resized, img_resized)
        factor /= 2.0
        return reg
    
    
    def forward(self, ir,rgb):
        phi_ir_m ,phi_rgb_m = self.stage_1(ir, rgb)
        ir_grid = self.transform_grid(ir,phi_ir_m)
        rgb_grid = self.transform_grid(rgb,phi_rgb_m)
        ir_warp_m = self.warping(ir, ir_grid)
        rgb_warp_m = self.warping(rgb, rgb_grid)
        reg_term_ir = self._calculate_regularization_term(phi_ir_m, ir_warp_m[0])
        reg_term_rgb = self._calculate_regularization_term(phi_rgb_m, rgb_warp_m[0])
        reg_term = (reg_term_ir + reg_term_rgb) / 2
        # phi_ir = self.ir_stage(ir_warp_m, rgb, phi_ir_m)
        # phi_rgb = self.rgb_stage(rgb_warp_m, ir, phi_ir_m)
        
        # return phi_ir,phi_ir_m, phi_rgb, phi_rgb_m
        return ir_warp_m, rgb_warp_m, rgb_grid, rgb_grid , reg_term


class MFlownet(nn.Module):
    
    def __init__(self,n_downs):
        super(MFlownet,self).__init__()
        self.FE_ir = nn.Sequential(Conv(1,8,3,1,1))
        self.FE_rgb = nn.Sequential(Conv(1,8,3,1,1))
        mun = 0
        mun = mun
        self.n_iters = n_downs
        
        downconvs = []
        upconvs_ir = []
        upconvs_rgb = []
        n_dim  = 16
        
        for i in range(n_downs):
            # n_dim = n_dim
            out_n_dim = 2 * n_dim if n_dim != 64 else 64
            downconvs += [DownConv(n_dim, out_n_dim,3,1,1,bias=True)]
            n_dim = out_n_dim
            
        self.downblock = nn.Sequential(*downconvs)
        
        self.resblock = nn.Sequential(Conv(out_n_dim, out_n_dim *2,1,1,0),ResnetBlock(out_n_dim*2, 3),
                                Conv(out_n_dim*2, out_n_dim, 1,1,0))
        
        sub_ndim = n_dim
        
        self.upconv_ir =  UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) 
        self.upconv_rgb = UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) 
        
        for i in range(n_downs-1):
            out_n_dim = sub_ndim //2 if sub_ndim > 8 else 8
            # upconvs_ir += [  UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) if (i == 0)  else Conv(out_n_dim, out_n_dim,3,1,1,bias=True)]
            upconvs_ir += [ Conv(sub_ndim, out_n_dim,3,1,1,bias=True)]

            
            sub_ndim = out_n_dim
        
        sub_ndim = n_dim
        for i in range(n_downs):
            out_n_dim = sub_ndim //2 if sub_ndim > 8 else 8
            # upconvs_rgb += [  UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) if (i == 0)  else Conv(out_n_dim, out_n_dim,3,1,1,bias=True)]
            upconvs_rgb += [ Conv(sub_ndim, out_n_dim,3,1,1,bias=True)]
            # if i > n_downs-4:
            
            sub_ndim = out_n_dim
            

        
        self.branch = CopyLayer()

        self.upblock_rgb = nn.Sequential(*upconvs_rgb)
        self.upblock_ir = nn.Sequential(*upconvs_ir)
        
        self.up_ir = UpConv(8,8,16,8,3,1,1)
        self.up_rgb = UpConv(8,8,16,8,3,1,1)
        
        self.refine_ir= nn.Sequential(ResnetBlock(8, 1),Conv(8,8, 1,1,0),Conv(8,2, 1, 1, 0,bias=True))
        self.refine_rgb = nn.Sequential(ResnetBlock(8, 1),Conv(8,8, 1,1,0),Conv(8,2, 1, 1, 0,bias=True))

        
    
    def forward(self, ir, rgb):
        fe_ir = self.FE_ir(ir)
        fe_rgb = self.FE_rgb(rgb)

        x = torch.cat([fe_ir, fe_rgb],1)

        fe = self.downblock(x)

        fe_re = self.resblock(fe)
        input_ir, input_rgb = self.branch(fe_re)

        
        out_ir = self.upconv_ir(input_ir, fe)
        out_rgb = self.upconv_rgb(input_rgb, fe)
        
        out_ir = self.upblock_ir(out_ir)
        out_rgb = self.upblock_rgb(out_rgb)
        
        out_ir = self.up_ir(out_ir, fe_rgb)
        out_rgb = self.up_rgb(out_rgb, fe_ir)
        
        out_ir = self.refine_ir(out_ir)
        out_rgb = self.refine_rgb(out_rgb)
        
        
        return out_ir, out_rgb
        
        
        
        # for i in range(self.n_iters):
            
        
        
class SingleBFlownet(nn.Module):
    
    def __init__(self,n_downs):
        super(MFlownet,self).__init__()
        self.FE_ir = nn.Sequential(Conv(1,8,3,1,1))
        self.FE_rgb = nn.Sequential(Conv(1,8,3,1,1))
        mun = 0
        mun = mun
        self.n_iters = n_downs
        
        downconvs = []
        upconvs_ir = []
        upconvs_rgb = []
        n_dim  = 16
        
        for i in range(n_downs):
            # n_dim = n_dim
            out_n_dim = 2 * n_dim if n_dim != 64 else 64
            downconvs += [DownConv(n_dim, out_n_dim,3,1,1,bias=True)]
            n_dim = out_n_dim
            
        self.downblock = nn.Sequential(*downconvs)
        
        self.resblock = nn.Sequential(Conv(out_n_dim, out_n_dim *2,1,1,0),ResnetBlock(out_n_dim*2, 3),
                                Conv(out_n_dim*2, out_n_dim, 1,1,0))
        
        sub_ndim = n_dim
        
        self.upconv_ir =  UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) 
        self.upconv_rgb = UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) 
        
        for i in range(n_downs-1):
            out_n_dim = sub_ndim //2 if sub_ndim > 8 else 8
            # upconvs_ir += [  UpConv(out_n_dim, out_n_dim, 2*out_n_dim, out_n_dim,3,1,1) if (i == 0)  else Conv(out_n_dim, out_n_dim,3,1,1,bias=True)]
            upconvs_ir += [ Conv(sub_ndim, out_n_dim,3,1,1,bias=True)]

            
            sub_ndim = out_n_dim
        
        self.branch = CopyLayer()

        self.upblock_rgb = nn.Sequential(*upconvs_rgb)
        self.upblock_ir = nn.Sequential(*upconvs_ir)
        
        self.up_ir = UpConv(8,8,16,8,3,1,1)
        
        self.refine_ir= nn.Sequential(ResnetBlock(8, 1),Conv(8,8, 1,1,0),Conv(8,2, 1, 1, 0,bias=True))

        
    
    def forward(self, ir, rgb):
        fe_ir = self.FE_ir(ir)
        fe_rgb = self.FE_rgb(rgb)

        x = torch.cat([fe_ir, fe_rgb],1)

        fe = self.downblock(x)

        fe_re = self.resblock(fe)
        input_ir, input_rgb = self.branch(fe_re)

        
        out_ir = self.upconv_ir(input_ir, fe)
        
        out_ir = self.upblock_ir(out_ir)
        
        out_ir = self.up_ir(out_ir, fe_rgb)

        out_ir = self.refine_ir(out_ir)
        
        return out_ir
        

    
    
    