import torch
import torch.nn as nn
import numpy as np
from copy import copy

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class skip_model(nn.Module):
    def __init__(self, num_layers, num_channels, num_output_channels, hidden_size, upsample_mode, act_fun,sig=None, bn_affine=True, skips=False,need_pad=True,need_last=True):
        super(skip_model, self).__init__()
        
        self.num_layers = num_layers
        #self.upsamp = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.act_fun = act_fun
        self.sig= sig
        self.skips = skips
        self.layer_inds = [] # record index of the layers that generate output in the sequential mode (after each BatchNorm)
        self.combinations = None # this holds input of the last layer which is upsampled versions of previous layers
        
        cntr = 1
        net1 = nn.Sequential()
        for i in range(num_layers-1):
            
            if need_pad:
                net1.add(nn.ReflectionPad2d(0))
            net1.add(nn.Conv2d(num_channels, num_channels, 1, 1, padding=0, bias=False))
            cntr += 1
            
            net1.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode,align_corners=True))
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
        if need_last:
            net2.add( nn.Conv2d(nic, num_channels, 1, 1, padding=0, bias=False) )
            net2.add(act_fun)
            net2.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            nic = num_channels
        if need_pad:
                net2.add(nn.ReflectionPad2d(0))
        net2.add(nn.Conv2d(nic, num_output_channels, 1, 1, padding=0, bias=False))
        if sig is not None:
                net2.add(self.sig)
        
        self.net1 = net1 
        self.net2 = net2
        
    def forward(self, x, scale_out=1):
        out1 = self.net1(x)
        if self.skips:
            intermed_outs = []
            for i,c in enumerate(self.net1):
                if i+1 in self.layer_inds:
                    f = self.net1[:i+1]
                    intermed_outs.append(f(x))
  
            intermed_outs = [self.up_sample(io,i+1) for i,io in enumerate(intermed_outs)]
           
            out1 = torch.cat(intermed_outs+[out1],1)
        self.combinations = copy(out1)
        out2 = self.net2(out1)
        return out2*scale_out
    def up_sample(self,img,layer_ind):
        if layer_ind != self.num_layers-1:
            samp_block = nn.Upsample(size=self.hidden_size[-1], mode=self.upsample_mode)#,align_corners=True)
            img = samp_block(img)
        return img

def skipdecoder(
        out_size = [256,256],
        in_size = [16,16],
        num_output_channels=3, 
        num_layers=6,
        num_channels=64,
        need_sigmoid=True,
        need_pad=False,
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        skips = True,
        nonlin_scales=False,
        need_last=True,
        ):
    
    
    scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
    if nonlin_scales:
        xscales = np.ceil( np.linspace(scale_x * in_size[0],out_size[0],num_layers-1) )
        yscales = np.ceil( np.linspace(scale_y * in_size[1],out_size[1],num_layers-1) )
        hidden_size = [(int(x),int(y)) for (x,y) in zip(xscales,yscales)]
    else:
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
    print(hidden_size)
    if need_sigmoid:
        sig = nn.Sigmoid()
    else:
        sig = None
    
    model = skip_model(num_layers, num_channels, num_output_channels, hidden_size,
                         upsample_mode=upsample_mode, 
                         act_fun=act_fun,
                         sig=sig,
                         bn_affine=bn_affine,
                         skips=skips,
                         need_pad=need_pad,
                         need_last=need_last,)
    return model