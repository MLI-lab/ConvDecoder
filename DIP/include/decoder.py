import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def deepdecoder(
        in_size,
        out_size,
        num_output_channels=3, 
        num_channels=[128]*5, 
        filter_size=1,
        need_sigmoid=True,
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bias=False,
        last_noup=False, # if true have a last extra conv-relu-bn layer without the upsampling before linearly combining them
        ):
    
    depth = len(num_channels)
    scale_x,scale_y = (out_size[0]/in_size[0])**(1./depth), (out_size[1]/in_size[1])**(1./depth)
    hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                    int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, depth)] + [out_size]
    
    print(hidden_size)
    
    num_channels = num_channels + [num_channels[-1],num_channels[-1]]
    
    n_scales = len(num_channels) 
    
    if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)) :
        filter_size   = [filter_size]*n_scales
    
    model = nn.Sequential()

    for i in range(len(num_channels)-2):
        model.add(conv( num_channels[i], num_channels[i+1],  filter_size[i], 1, pad=pad, bias=bias))
        if upsample_mode!='none' and i != len(num_channels)-2:
            # align_corners: from pytorch.org: if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default: False
            # default seems to work slightly better
            model.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode,align_corners=False))
        
        if(bn_before_act): 
            model.add(nn.BatchNorm2d( num_channels[i+1] ,affine=bn_affine))
        if act_fun is not None:    
            model.add(act_fun)
        if not bn_before_act:
            model.add(nn.BatchNorm2d( num_channels[i+1], affine=bn_affine))
    
    if last_noup:
        model.add(conv( num_channels[-2], num_channels[-1],  filter_size[-2], 1, pad=pad, bias=bias))
        model.add(act_fun)
        model.add(nn.BatchNorm2d( num_channels[-1], affine=bn_affine))
    
    model.add(conv( num_channels[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
   
    return model





def decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True,
        need_tanh=False,
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad, bias=bias))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad,bias=bias))        
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())
   
    return model





##########################
"""

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)




def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


class skip_model(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, upsample_mode, act_fun,sig=None, bn_affine=True, skips=False):
        super(skip_model, self).__init__()
        
        self.num_layers = num_layers
        self.upsamp = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.act_fun = act_fun
        self.sig= sig
        self.skips = skips
        self.layer_inds = [] # record index of the layers that generate output in the sequential mode (after each BatchNorm)
        
        cntr = 1
        net1 = nn.Sequential()
        for i in range(num_layers-1):
        
            net1.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            cntr += 1
            
            net1.add(nn.Conv2d(num_channels, num_channels, 1, 1, padding=0, bias=False))
            cntr += 1
            
            net1.add(act_fun)
            cntr += 1
            net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            if i != num_layers - 2:
                self.layer_inds.append(cntr)
            cntr += 1
                
        net2 = nn.Sequential()
        nic = num_channels
        if skips:
            nic = num_channels*(num_layers-1)
        net2.add(nn.Conv2d(nic, num_output_channels, 1, 1, padding=0, bias=False))
        if sig is not None:
                net2.add(self.sig)
        
        self.net1 = net1 
        self.net2 = net2

    def forward(self, x):
        out1 = self.net1(x)
        if self.skips:
            intermed_outs = []
            for i,c in enumerate(self.net1.children()):
                if i+1 in self.layer_inds:
                    f = self.net1[:i+1]
                    intermed_outs.append(f(x))
  
            intermed_outs = [self.up_sample(io,i+1) for i,io in enumerate(intermed_outs)]
           
            out1 = torch.cat(intermed_outs+[out1],1)
            
        out2 = self.net2(out1)
        return out2
    def up_sample(self,img,layer_ind):
        loop = self.num_layers - 1 - layer_ind
        for i in range(loop):
            img = self.upsamp(img)
        return img

def skipdecoder(
        num_output_channels=3, 
        num_layers=6,
        num_channels=64,
        need_sigmoid=True, 
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        skips = True,
        ):
    
    if need_sigmoid:
        sig = nn.Sigmoid()
    else:
        sig = None
    model = nn.Sequential()
    model.add(skip_model(num_layers, num_channels, num_output_channels, 
                         upsample_mode=upsample_mode, 
                         act_fun=act_fun,
                         sig=sig, bn_affine=bn_affine, skips=skips))
    return model
"""

