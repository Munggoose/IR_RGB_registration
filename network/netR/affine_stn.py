import torch
import torch.nn.functional as F
from torch import nn
from .block_org import DownBlock,Conv

class AffineNetwork(nn.Module):
    """Builds an network that predicts the 6 parameters ued in a affine transformation.

    The network follow a conventional encoder CNN with fully connected layers at the end. You may define different
    network configurations by adding or modifying existing configurations (e.g 'A').

    For example - using configuration 'A' will build the following network:

    Convs:
    Block1 : 3x3 Conv (32 fltrs) -> InstanceNorm2D -> ReLU -> 2x2 Pooling
    Block2 : 3x3 Conv (64 fltrs) -> InstanceNorm2D -> ReLU -> 2x2 Pooling
    Block3 : 3x3 Conv (128 fltrs) -> InstanceNorm2D -> ReLU -> 2x2 Pooling
    Block4 : 3x3 Conv (256 fltrs) -> InstanceNorm2D -> ReLU -> 2x2 Pooling
    Block5 : 3x3 Conv (256 fltrs) -> InstanceNorm2D -> ReLU -> 2x2 Pooling
    Localization:
    L1 : Linear (256 output neurons)-> ReLU
    L2 : Linear (6 output neurons) <<--NOTE--<< This layer is initialized to zeros.
    """

    def __init__(self, in_channels_a, in_channels_b, height, width, init_func='kaiming'):
        """

        :param in_channels_a: channels used for modality A
        :param in_channels_b: channels used for modality B
        :param height: image height
        :param width: image width
        :param cfg: the network configurations
        :param init_func: the initialization method used to initialize the Convolutional layers weights.
        """
        super(AffineNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = height, width

        nconvs = 5
        convs = []
        prev_nf = in_channels_a + in_channels_b
        nf = 32
        
        for _ in range(nconvs):
            convs.append(DownBlock(prev_nf, nf, 3, 1, 1, bias=True, activation='relu',
                                   init_func=init_func, use_norm=True,
                                   use_resnet=False,
                                   skip=False,
                                   refine=False,
                                   pool=True))
            prev_nf = nf
            nf = min(2 * nf, 256)
        
        self.convs = nn.Sequential(*convs)
        act = nn.ReLU(inplace=True) # get_activation(activation=cfg_activation[cfg])
        self.local = nn.Sequential(
            nn.Linear(prev_nf * (self.h // (2 ** nconvs)) * (self.w // (2 ** nconvs)), nf, bias=True),
            act,
            nn.Linear(nf, 6, bias=True))

        # Start with identity transformation
        self.local[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.local[-1].bias.data.zero_()

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        dtheta = self.local(x)
        return dtheta


class AffineSTN(nn.Module):
    """This class is generates and applies the affine transformation on the input images"""

    def __init__(self, cfg):
        super(AffineSTN, self).__init__()
        self.cfg = cfg
        height =self.cfg.img_size[0] 
        width = self.cfg.img_size[1]

        nc_a = cfg.in_ch
        nc_b = cfg.in_ch

        init_func = 'kaiming'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AffineNetwork(nc_a, nc_b, height, width, init_func)
        self.identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(self.device)


    def _get_theta(self, img_a, img_b):
        """Predict the 3x2 parameters that define the affine transformation."""
        bsize = img_a.size(0)
        dtheta = self.net(img_a, img_b)
        theta = dtheta + self.identity_theta.unsqueeze(0).repeat(bsize, 1)
        return theta


    def get_grid(self, img_a, img_b):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        theta = self._get_theta(img_a, img_b)
        resampling_grid = F.affine_grid(theta.view(-1, 2, 3), img_a.size())
        return resampling_grid


    def warping(self, x, resampling_grid):
        warped_images = F.grid_sample(x, resampling_grid, mode='bilinear', padding_mode='zeros',
                                                align_corners=False)
        return warped_images

    def forward(self, img_a, img_b, apply_on=None):
        """
        Predicts the spatial alignment needed to align img_a with img_b. The spatial transformation will be applied on
        the tensor passed by apply_on (if apply_on is None then the transformation will be applied on img_a).

        :param img_a: the source image.
        :param img_b: the target image.
        :param apply_on: the geometric transformation can be applied on different tensors provided by this list.
                If not set, then the transformation will be applied on img_a.
        :return: a list of the warped images (matching the order they appeard in apply on), and the regularization term
                calculated for the predicted transformation.
        """
        # Get Affine transformation
        dtheta = self.net(img_a, img_b)
        theta = dtheta + self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)
        # Wrap image wrt to the deformation field
        if apply_on is None:
            apply_on = [img_a]
        warped_images = []

        for img in apply_on:
            resampling_grid = F.affine_grid(theta.view(-1, 2, 3), img.size())
            warped_images.append(
                F.grid_sample(img, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False))
        # Calculate STN regularization term - for affine transformation, the predicted affine transformation should not
        # largely deviate from the identity transformation.
        reg_term = self._calculate_regularization_term(dtheta)
        return warped_images[0], resampling_grid, reg_term

    def _calculate_regularization_term(self, theta):
        x = torch.mean(torch.abs(theta))
        return x
    
    
    