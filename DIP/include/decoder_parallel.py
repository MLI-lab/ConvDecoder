import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class cat_model(nn.Module):
    def __init__(self, decoders_numlayers_list,decoders_last_channels,num_channels, num_output_channels,
                 upsample_mode,act_fun,hidden_size,sig=None,bn_affine=True,need_pad=True):
        super(cat_model, self).__init__()
        
        self.sig = sig
        nets = []
        M = max(decoders_numlayers_list)
        for n,num_layers in enumerate(decoders_numlayers_list):
            nc = num_channels
            net = nn.Sequential()
            for i in range(num_layers-1):
                if need_pad:
                    net.add(nn.ReflectionPad2d(0))
                net.add(nn.Conv2d(num_channels, nc, 1, 1, padding=0, bias=False))
                net.add(nn.Upsample(size=hidden_size[n][i], mode=upsample_mode))
                net.add(act_fun)
                net.add(nn.BatchNorm2d( nc, affine=bn_affine))
                    
            nc = decoders_last_channels[n]
            if need_pad:
                net.add(nn.ReflectionPad2d(0))
            net.add(nn.Conv2d(num_channels, nc, 1, 1, padding=0, bias=False))
            if self.sig is not None:
                net.add(self.sig)
            
            nets.append(net)
            del(net)
        
        self.net1 = nets[0]
        self.net2 = nets[1]
        self.net3 = nets[2]
        self.norm = nn.BatchNorm2d( sum(decoders_last_channels), affine=bn_affine)
        self.last_conv = nn.Conv2d(sum(decoders_last_channels),num_output_channels,1,1,padding=0,bias=False)

    def forward(self,x,scale_out=1):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        
        last_inp = torch.cat([out1,out2,out3],1)
        out = self.last_conv(self.norm(last_inp))
        if self.sig is not None:
            out = self.sig(out)
        return out*scale_out
def pardecoder(out_size = [256,256],
        in_size = [16,16],
        num_output_channels=3, 
        num_channels=128,
        decoders_numlayers_list = [2,4,6], # (ascending order) determines the number of layers per each decoder in the parallel structure
        decoders_last_channels = [20,20,20], # last layer channel contribution of each decoder
        need_sigmoid=True, 
        need_pad=True, 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_affine = True,
        nonlin_scales=False,
        ):
    
    hidden_size = []
    for num_layers in decoders_numlayers_list:
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
        if nonlin_scales:
            xscales = np.ceil( np.linspace(scale_x * in_size[0],out_size[0],num_layers-1) )
            yscales = np.ceil( np.linspace(scale_y * in_size[1],out_size[1],num_layers-1) )
            h_s = [(int(x),int(y)) for (x,y) in zip(xscales,yscales)]
        else:
            scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
            h_s = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
        hidden_size.append(h_s)
    print(hidden_size)
    if need_sigmoid:
        sig = nn.Sigmoid()
    else:
        sig = None
    
    model = cat_model(decoders_numlayers_list,
                      decoders_last_channels,
                      num_channels, 
                      num_output_channels,
                      upsample_mode,
                      act_fun,
                      hidden_size,
                      sig = sig,
                      bn_affine = bn_affine,
                      need_pad=need_pad,)
   
    return model