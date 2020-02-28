import torch
import torch.nn as nn

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
    model = nn.Sequential()
    model.add(skip_model(num_layers, num_channels, num_output_channels, 
                         upsample_mode=upsample_mode, 
                         act_fun=act_fun,
                         sig=sig, bn_affine=bn_affine, skips=skips))
    return model