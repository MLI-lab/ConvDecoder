import torch
import torch.nn as nn
import numpy as np
from copy import copy

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class conv_model(nn.Module):
    def __init__(self, num_layers, strides, num_channels, num_output_channels, hidden_size, upsample_mode, act_fun,sig=None, bn_affine=True, skips=False,intermeds=None,bias=False,need_lin_comb=False,need_last=False,kernel_size=3):
        super(conv_model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.act_fun = act_fun
        self.sig= sig
        self.skips = skips
        self.intermeds = intermeds
        self.layer_inds = [] # record index of the layers that generate output in the sequential mode (after each BatchNorm)
        self.combinations = None # this holds input of the last layer which is upsampled versions of previous layers
        
        cntr = 1
        net1 = nn.Sequential()
        for i in range(num_layers-1):
            
            net1.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))#,align_corners=True))
            cntr += 1
            
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=bias)
            net1.add(conv)
            cntr += 1
            
            #net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            net1.add(act_fun)
            cntr += 1
            
            if need_lin_comb:
                net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine)) 
                #net1.add(act_fun)
                cntr += 1

                net1.add(nn.Conv2d(num_channels, num_channels, 1, 1, padding=0, bias=bias))
                cntr += 1

                #net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
                net1.add(act_fun)
                cntr += 1
            
            #net1.add(act_fun)
            net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            if i != num_layers - 2: # penultimate layer will automatically be concatenated if skip connection option is chosen
                self.layer_inds.append(cntr)
            cntr += 1

        net2 = nn.Sequential()
        
        nic = num_channels
        if skips:
            nic = num_channels*( sum(intermeds)+1 )
        
        if need_last:
            net2.add( nn.Conv2d(nic, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=bias) )
            net2.add(act_fun)
            net2.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            nic = num_channels
            
        net2.add(nn.Conv2d(nic, num_output_channels, 1, 1, padding=0, bias=bias))
        
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
            intermed_outs = [intermed_outs[i] for i in range(len(intermed_outs)) if self.intermeds[i]]
            intermed_outs = [self.up_sample(io) for io in intermed_outs]
            out1 = torch.cat(intermed_outs+[out1],1)
        self.combinations = copy(out1)
        out2 = self.net2(out1)
        return out2*scale_out
    def up_sample(self,img):
        samp_block = nn.Upsample(size=self.hidden_size[-1], mode=self.upsample_mode)#,align_corners=True)
        img = samp_block(img)
        return img

def convdecoder(
        out_size = [256,256],
        in_size = [16,16],
        num_output_channels=3,
        num_layers=6,
        strides=[1]*6,
        num_channels=64,
        need_sigmoid=True, 
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        skips = True,
        intermeds=None,
        nonlin_scales=False,
        bias=False,
        need_lin_comb=False,
        need_last=False,
        kernel_size=3,
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
        #sig = nn.Tanh()
        #sig = nn.Softmax()
    else:
        sig = None
    
    model = conv_model(num_layers, strides, num_channels, num_output_channels, hidden_size,
                         upsample_mode=upsample_mode, 
                         act_fun=act_fun,
                         sig=sig,
                         bn_affine=bn_affine,
                         skips=skips,
                         intermeds=intermeds,
                         bias=bias,
                         need_lin_comb=need_lin_comb,
                         need_last = need_last,
                         kernel_size=kernel_size,)
    return model