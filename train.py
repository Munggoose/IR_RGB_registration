from Dataset.OzrayDataset import DefaultDataset,SegmentDataset
from Dataset.FlirDatset import FLIRDataset
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn

from network.BcascadeDeformation import BranchCascadeRegistrator
from network.SingleDeformation import SingleDeformation
from network.orgDeformation import OrgSingleDeformation
from network.two_stage_Deformation import TwoStageDeformationGenerator
from utils.visualizer import MMIR_Visualer
from tqdm import tqdm

from utils.metric import calcualte_IOU

import numpy as  np
import random

def baseparser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', type=int, default=1)
    
    parser.add_argument('-e','--n_epoch',type=int, default=10)
    parser.add_argument('-ee','--eval_epoch',type=int, default=5)
    parser.add_argument('-d','--debug',action='store_true')
    parser.add_argument('-i_size','--img_size',type=list, default = [512,512])
    parser.add_argument('-device','--device',type=str, default='cuda')
    parser.add_argument('-lr',type=float, default=1e-4)
    parser.add_argument('-v','--visualizer', action='store_true')
    parser.add_argument('-s','--seed',type=int, default=1)
    #model
    parser.add_argument('-m','--model',type=str,default='Bcascade')
    
    #dataset
    parser.add_argument('--root',type=str, default='F:\\data/ozray_data/0406')
    # parser.add_argument('--root',type=str, default='F:\\data/ozray_data/segdata/train')
    parser.add_argument('-bs','--batch_size',type=int, default=2)
    parser.add_argument('-in_ch','--in_ch',type=int, default=1)
    parser.add_argument('-out_ch','--out_ch',type=int, default=1)
    
    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for the GAN loss.')
    parser.add_argument('--lambda_recon', type=float, default=100.0,
                            help='Weight for the L1 reconstruction loss.')
    parser.add_argument('--lambda_smooth', type=float, default=0.0, help='Regularization term used by the STN')

    return parser.parse_args()
    # parser.add_arugment('--grey',)
    

class Trainner:
    
    def __init__(self,cfg):
        self.train
        self.cfg = cfg
        # trainset = DefaultDataset(self.cfg.root, img_size=self.cfg.img_size, rgb=False)
        # self.trainloader = DataLoader(trainset,batch_size=self.cfg.batch_size, shuffle=True)
        # testset = DefaultDataset(self.cfg.root, img_size=self.cfg.img_size,train=False, rgb=False)
        # self.testloader = DataLoader(testset,batch_size=self.cfg.batch_size, shuffle=True)
        
        
        # trainset = FLIRDataset(self.cfg.root, img_size=self.cfg.img_size, rgb=False)
        # self.trainloader = DataLoader(trainset,batch_size=self.cfg.batch_size, shuffle=True)
        # testset = FLIRDataset(self.cfg.root, img_size=self.cfg.img_size,train=False, rgb=False)
        # self.testloader = DataLoader(testset,batch_size=self.cfg.batch_size, shuffle=True)
        
        trainset = SegmentDataset(self.cfg.root, img_size=self.cfg.img_size, rgb=False)
        self.trainloader = DataLoader(trainset,batch_size=self.cfg.batch_size, shuffle=True)
        testset = SegmentDataset(self.cfg.root, img_size=self.cfg.img_size,train=False, rgb=False)
        self.testloader = DataLoader(testset,batch_size=self.cfg.batch_size, shuffle=True)
        
        
        if cfg.model =='S':
            self.model = SingleDeformation(cfg)
        elif cfg.model =='M':
            self.model = BranchCascadeRegistrator(cfg)
        elif cfg.model =='O':
            self.model = OrgSingleDeformation(cfg)
        elif cfg.model == 'T':
            self.model =TwoStageDeformationGenerator(cfg)
            
        else:
            assert f"no model type {cfg.model}"
        
        self.model.cuda()
        self.visualizer = self.cfg.visualizer

        if self.visualizer:
            
            run_name = f"exp_{self.cfg.seed}_{self.cfg.model}_{self.cfg.lr}_{self.cfg.lambda_recon}"
            project = "RGB_IR_registration"
            self.vis = MMIR_Visualer(self.model,run_name,project,cfg)
    
    
    def train(self):
        self.model.train()
        epoch_bar = tqdm(range(1,self.cfg.n_epoch+1))
        for epoch in epoch_bar:
            D_l = []
            G_l = []
            ir_rec_l = []
            rgb_rec_l = []
            
                        
            for rgb, ir in tqdm(self.trainloader):
                
                rgb = rgb.cuda()
                ir = ir.cuda()

                warped_ir, warped_rgb = self.model(ir, rgb)
                D_loss, gan_loss, rec_ir_loss, rec_rgb_loss= self.model.optimize_parameters()
                
                D_l.append(D_loss.item())
                G_l.append(gan_loss.item())
                ir_rec_l.append(rec_ir_loss.item())
                
                rgb_rec_l.append(rec_rgb_loss.item())
                if self.cfg.debug:
                    break
            
            n_datas = len(self.trainloader)
            avg_ir_recs = sum(ir_rec_l) / n_datas
            avg_rgb_recs = sum(rgb_rec_l) / n_datas
            avg_G_l =  sum(G_l) / n_datas
            avg_D_l =  sum(D_l) / n_datas
        
            
            if self.visualizer:
                alpha_img = self.vis.alpha_blending(self.model.warped_ir[0],self.model.rgb[0])
                alpha_img = self.vis.toWandBimg(alpha_img,f'output {epoch}')
                warped_img = self.vis.toWandBimg(self.model.warped_ir[0],f'wapred_img' )
                
                input_img = self.vis.alpha_blending(self.model.ir[0],self.model.rgb[0])
                input_img = self.vis.toWandBimg(input_img,f'input {epoch}')
                
                self.vis.updatelog('result_img',[input_img,warped_img,alpha_img])
                self.vis.updatelog('L_ir_rec',avg_ir_recs)
                self.vis.updatelog('L_rgb_rec',avg_rgb_recs)
                self.vis.updatelog('L_G',avg_G_l)
                self.vis.updatelog('L_D',avg_D_l)
                
            if epoch % self.cfg.eval_epoch == 0:
                self.evaluation()
                self.model.train()
                
            if self.visualizer:
                self.vis.updateWandb()
    
    def evaluation(self):
        self.model.eval()
        ious = []
        mask_losses = []
        # mask_criterion = nn.L1Loss(reduction='sum')
        for rgb, ir,rgb_mask, ir_mask in tqdm(self.testloader):
            rgb = rgb.cuda()
            ir = ir.cuda()
            ir_mask = ir_mask.cuda()
            rgb_mask = rgb_mask.cuda()
            _, _ = self.model(ir, rgb)
            
            warped_ir_mask_affine = self.model.netAffine.warping(ir_mask ,self.model.homography)
            warped_ir_mask  = self.model.netDFG.warping(warped_ir_mask_affine, self.model.ir_grid)

            iou = calcualte_IOU(warped_ir_mask,rgb_mask)
            ious.append(iou)
            
            # mask_loss = mask_criterion(warped_ir_mask,rgb_mask)
            # mask_losses.append(mask_loss.item())
            # warp_ir = image_warping_by_homography(ir,weight_f,w,h)
            
        iou_score = np.array(ious).mean()
        # mask_l1_loss = np.array(mask_losses).mean()
        print(f"IOU_SCORE: {iou_score}")# - Mask_L1: {mask_l1_loss}
            
        if self.visualizer:   
            self.vis.updatelog('IoU',iou_score)
            # self.vis.updatelog('Mask_L1',mask_l1_loss)
            
            
    
    
    def infer(sefl):
        pass
    
    
    
    

if __name__=='__main__':
    
    cfg = baseparser()
    # model = BranchCascadeRegistrator(cfg)
    for t in range(19,cfg.loop):
        cfg.seed = t
        
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        Tm = Trainner(cfg)
        Tm.train()
    
    pass

