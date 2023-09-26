import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.seg_mask import load_seg_pt, shape_to_mask

import os
from PIL import Image
from glob import glob 
import torchvision
from abc import ABC
from abc import abstractmethod



class DefaultDataset(Dataset):
    #Default setting for OzrayDataset
    
    def __init__(self, root, img_size=[256,256], transform = None , rgb=True,train=True,grey=True):
        super().__init__()
        
        self.grey= grey
        self.train = train
        
        if train:
            self.root = os.path.join(root, "train") # 나중에 Train 데이터용으로 변경
        else:
            self.root = os.path.join(root, "test")
            self.mask_root = os.path.join(self.root, 'mask')
        self.rgb_ch = rgb
        

        self.rgb_root = os.path.join(self.root, "rgb")
        self.ir_root = os.path.join(self.root, "ir")
        
        if transform:
            self.transform = transforms.Compose(transform.append(transforms.Resize(img_size)))
        
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(img_size)])
        
        self.rgb_list = glob(os.path.join(self.rgb_root, '*.png'))
    
    
    def make_label(self, rgb_path):
        
        rgb = Image.open(rgb_path)
        rgb_name = os.path.basename(rgb_path)
        
        ir_name = os.path.basename(rgb_name)[:-7] + 'ir.png'
        ir_path = os.path.join(self.ir_root,ir_name)
        ir = Image.open(ir_path)
        
        if not self.train:
            rgb_mask_path = os.path.join(self.mask_root, rgb_name)
            rgb_mask = Image.open(rgb_mask_path)
            rgb_mask_path = os.path.join(self.mask_root, rgb_name)
            rgb_mask = Image.open(rgb_mask_path)
            ir_mask_path = os.path.join(self.mask_root,ir_name)
            ir_mask = Image.open(ir_mask_path)


        if self.grey:
            ir = ir.convert('L')
            rgb = rgb.convert('L')
        
        if self.train:
            return self.transform(rgb),self.transform(ir)
             
        else:
            return  self.transform(rgb),self.transform(ir),\
                    self.transform(rgb_mask),self.transform(ir_mask)

    
    def __getitem__(self, index):
        # return self.load_img(self.rgb_list[index])
        return self.make_label(self.rgb_list[index])
    
    
    def __len__(self):
        return len(self.rgb_list)
    


class SegmentDataset(Dataset):
    #Default setting for OzrayDataset
    
    def __init__(self, root, img_size=[256,256], transform = None , rgb=True,train=True,grey=True):
        super().__init__()
        
        self.grey= grey
        self.train = train
        
        if train:
            self.root = os.path.join(root, "train") # 나중에 Train 데이터용으로 변경
        else:
            self.root = os.path.join(root, "test")
            # self.mask_root = os.path.join(self.root, 'mask')
        self.rgb_ch = rgb
        

        self.rgb_root = os.path.join(self.root, "rgb")
        self.ir_root = os.path.join(self.root, "ir")
        
        if transform:
            self.transform = transforms.Compose(transform.append(transforms.Resize(img_size)))
        
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(img_size)])
        
        self.rgb_list = glob(os.path.join(self.rgb_root, '*.png'))
    
    
    def make_label(self, rgb_path):
        
        rgb = Image.open(rgb_path)
        rgb_name = os.path.basename(rgb_path)

        
        ir_name = os.path.basename(rgb_name)[:-7] + 'ir.png'
        ir_path = os.path.join(self.ir_root,ir_name)
        ir = Image.open(ir_path)
        
        if not self.train:
            rgb_json = rgb_path[:-3] + 'json'
            rgb_pts = load_seg_pt(rgb_json)
            rgb_mask = shape_to_mask(rgb.size,rgb_pts)
            
            ir_json = ir_path[:-3] + 'json'
            ir_pts = load_seg_pt(ir_json)
            ir_mask = shape_to_mask(ir.size,ir_pts)
                    

        if self.grey:
            ir = ir.convert('L')
            rgb = rgb.convert('L')
        
        if self.train:
            return self.transform(rgb),self.transform(ir)
             
        else:
            return  self.transform(rgb),self.transform(ir),\
                    self.transform(rgb_mask),self.transform(ir_mask)

    
    def __getitem__(self, index):
        # return self.load_img(self.rgb_list[index])
        return self.make_label(self.rgb_list[index])
    
    
    def __len__(self):
        return len(self.rgb_list)