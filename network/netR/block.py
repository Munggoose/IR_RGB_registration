import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import partial


class BaseBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.last_layer = False
            
    def init_weight(self, module):
        init_ = partial(nn.init.kaiming_normal_, a=0.0, nonlinearity='relu', mode='fan_in')
        class_name = self.__class__.__name__
        if class_name.find('Conv') != -1:
            init_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if class_name.find('BatchNorm2d') != -1:
            nn.init.normal_(module.weight.data,0.0,1.0)
            nn.init.constant_(module.bias.data, 0.0)
        
        if self.last_layer:
            init_ = nn.init.kaiming_normal_(a=0.0, nonlinearity='zero', mode='fan_in')
            init_(self.projection.weight)



class Conv(nn.Module):
    
    def __init__(self, in_ch, out_ch, k_size, stride, padding, resnet_n_blocks=0, bias=True, activation='relu',
                 use_norm=False):
        super(Conv, self).__init__()
        
        self.conv2d = nn.Conv2d(in_ch, out_ch, k_size,stride,padding, bias=bias)
        self.resnet_block = ResnetBlock(out_ch, resnet_n_blocks) if resnet_n_blocks != 0 else None
        self.norm = nn.InstanceNorm2d(out_ch, affine=False, track_running_stats=False) if use_norm else None
        self.activation = nn.LeakyReLU(negative_slope= 0.2, inplace=False)
        
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)
    
    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x



class DownConv(BaseBlock):
    def __init__(self,in_ch, out_ch, k_size, stride, padding,bias=True,use_resnet =False,use_norm=True,init_func='kaiming',
                    skip=False,pool=False, pool_size=2):
        super().__init__()
        self.init_func = init_func
        
        # self.conv_0 = nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=bias)
        # self.norm_0 = nn.InstanceNorm2d(out_ch, affine=False, track_running_stats=False) if use_norm  else None
        # self.activation_0 = nn.ReLU()#get_activation(activation)
        
        self.conv_0 = Conv(in_ch,out_ch,k_size, stride, padding, bias=bias,use_resnet=use)

        self.conv_1 = nn.Conv2d(out_ch, out_ch, k_size, stride, padding, bias=bias)
        self.norm_1 = nn.InstanceNorm2d(out_ch, affine=False, track_running_stats=False) if use_norm  else None
        self.activation_1 = nn.ReLU()#get_activation(activation)
        
        # self.apply(self.init_weight)
        
        self.skip = skip
        self.pool = None
        
        if pool:
            self.pool  = nn.MaxPool2d(kernel_size=pool_size)
    
    def forward(self, x):

        
        x = skip = self.conv_0(x)
        
        
        if self.pool is not None:
            x = self.pool(x)
            
        x = skip = self.conv_1(x)
        
        if self.skip:
            return x, skip
        else:
            return x



class UpConv(BaseBlock):
    
    def __init__(self,nc_down_stream,nc_skip_stream,nc_hidden,nc_out ,k_size, stride, padding,bias=True,
                    refine=False,use_attention=False,last_layer=False ):
        super().__init__()
        self.conv_0 = nn.Conv2d(nc_down_stream+nc_skip_stream, nc_hidden, k_size,stride, padding,bias=bias)
        self.act_0 = nn.ReLU()#get_activation(activation)
        self.conv_1 = nn.Conv2d(nc_hidden, nc_hidden, k_size, stride, padding,bias=bias)
        self.act_1 = nn.ReLU()#get_activation(activation)
        self.last_layer = last_layer
        
        self.refine = None
        if refine:
            self.refine = nn.Conv2d(nc_hidden, nc_hidden, k_size, stride, padding, bias=bias )
            self.act_2 = nn.ReLU()#get_activation(activation)
            # self.norm_0 = nn.InstanceNorm2d(nc_hidden, affine=True, track_running_stats=False)
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_gate = AttentionGate()
        
        self.up_conv = nn.Conv2d(nc_hidden, nc_out,1, 1, 0, bias=bias)
        self.up_act = nn.ReLU() #get_activation(activation)
        
        if self.last_layer:
            self.projection = nn.Conv2d(nc_out, 2, kernel_size=1, stride=1, padding=0, bias= bias)
        
        # self.apply(self.init_weight)
    
    
    def forward(self, main_stream, skip_stream):
        main_stream_size = main_stream.size()
        skip_stream_size = skip_stream.size()
        
        assert main_stream_size == skip_stream_size, "Diffrent skip stream size and mainstreaam size"
        
        if self.use_attention:
            skip_stream = self.attention_gate(main_stream, skip_stream)
        
        x = torch.cat([main_stream, skip_stream], 1)
        x = self.act_0(self.conv_0(x))
        x = self.act_1(self.conv_1(x))
        
        if self.refine:
            x = self.refine(x)
        
        if self.last_layer:
            x = self.projection(x) + main_stream
        else:
            x = self.up_conv(x)
            x = self.up_act(x)
        
        return x


class ResnetLayer(nn.Module):
    
    def __init__(self, dim, padding_type, use_bias):
        super().__init__()
        blocks = []
        padding = 0
        if padding_type =='reflect':
            blocks += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding=1
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')
        
        blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias), 
                    nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),]
        
        self.net = nn.Sequential(*blocks)
        
    def forward(self, x):
        """ Skip connection forward"""
        out = x + self.net(x)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, n_blocks):
        super(ResnetBlock, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetLayer(dim, padding_type='reflect', use_bias=True)]
        self.model = nn.Sequential(*model)
        init_ = partial(nn.init.kaiming_normal_,a=0.0, nonlinearity='relu', mode='fan_in')
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
                
        self.model.apply(init_weights)
    
    
    def forward(self, x):
        return self.model(x)



class AttentionGate(nn.Module):
    def __init__(self, nc_g, nc_x, nc_hidden, use_norm=False, init_func='kaiming', mask_channel_wise=False):
        super(AttentionGate, self).__init__()
        
        self.conv_g = nn.Conv2d(nc_g, nc_hidden, 1,1,0,bias=True)
        # self.norm_g = nn.InstanceNorm2d(nc_hidden, affine=False, track_running_stats=False1)
        self.conv_x = nn.Conv2d(nc_x, nc_hidden, 1,1,0,bias=True)
        # self.norm_g = nn.InstanceNorm2d(nc_hidden, affine=False, track_running_stats=False1)
        
        self.residual = nn.ReLU(inplace=True)
        self.mask_channel_wise = mask_channel_wise
        
        self.acttetnion_map = nn.Sequential(nn.Conv2d(nc_hidden , nc_x if mask_channel_wise else 1,1,1,0, bias =True),
                                            nn.Sigmoid())
        
    def init_weight(self, module):
        init_ = nn._(self.init_func)
        class_name = self.__class__.__name__
        if class_name.find('Conv') != -1:
            init_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if class_name.find('BatchNorm2d') != -1:
            nn.init.normal_(module.weight.data,0.0,1.0)
            nn.init.constant_(module.bias.data, 0.0)
        
        if self.last_layer:
            init_ = nn.init.kaiming_normal_(a=0.0, nonlinearity='zero', mode='fan_in')
            init_(self.projection.weight)


    def forward(self, g, x):
        x_size = x.size()
        g_size = g.size()
        x_resized = x
        g_c = self.conv_g(g)
        x_c = self.conv_x(x_resized)
        if x_c.size(2) != g_size[2] and x_c.size(3) != g_size[3]:
            x_c = F.interpolate(x_c, (g_size[2], g_size[3]), mode='bilinear', align_corners=False)
        combined = self.residual(g_c + x_c)
        alpha = self.attention_map(combined)
        if not self.mask_channel_wise:
            alpha = alpha.repeat(1, x_size[1], 1, 1)
        alpha_size = alpha.size()
        if alpha_size[2] != x_size[2] and alpha_size[3] != x_size[3]:
            alpha = F.interpolate(x, (x_size[2], x_size[3]), mode='bilinear', align_corners=False)
            
        return alpha * x
    

class CopyLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return x , x