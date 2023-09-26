from abc import ABC
from abc import abstractmethod
import wandb
from collections import OrderedDict

class BaseVisualizer(ABC):
        
    def __init__(self, model, run_name,project,args):
        wandb.init(project=project, entity="munggoose",reinit=True)
        wandb.run.name = run_name
        
        if args:
            wandb.config.update(args)
            
        wandb.watch(model)
        self.log_dict = OrderedDict()
        
    def toWandBimg(self, img_tensor, caption):
        if img_tensor.is_cuda:
            img_tensor = img_tensor.to('cpu')
        return wandb.Image(img_tensor.detach(), caption=caption)
    
    def to_cpu(self, val_tensor):
        if val_tensor.is_cuda:
            val_tensor.to('cpu')
        
        return val_tensor    
            
    @abstractmethod
    def updatelog(self,key, val):
        pass
        # self.log_dict[key] = val
    
    # @Munggoose
    def updateWandb(self):
        wandb.log(self.log_dict)
    


class MMIR_Visualer(BaseVisualizer):
    
    def updatelog(self,key,val):
        self.log_dict[key] = val
    
    def alpha_blending(self,img_a,img_b):
        img_a = self.to_cpu(img_a)
        img_b = self.to_cpu(img_b)
        
        img_blending = img_a * 0.5 + 0.5 * img_b
        
        return img_blending
    
    