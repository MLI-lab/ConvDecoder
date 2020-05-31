import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class catc_model(nn.Module):
    def __init__(self, decoders_numlayers_list,decoders_last_channels,num_channels, num_output_channels,
                 upsample_mode,act_fun,hidden_size,sig=None,bn_affine=True,bias=True,need_lin_comb=False,
                 need_last=False,kernel_size=[3]*3):
        super(catc_model, self).__init__()
        
        self.sig = sig
        nets = []
        M = max(decoders_numlayers_list)
        for n,num_layers in enumerate(decoders_numlayers_list):
            nc = num_channels
            net = nn.Sequential()
            for i in range(num_layers-1):
                net.add(nn.Upsample(size=hidden_size[n][i], mode=upsample_mode))
                net.add(nn.Conv2d(num_channels, nc, kernel_size[n], 1, padding=(kernel_size[n]-1)//2, bias=bias))
                net.add(nn.BatchNorm2d( nc, affine=bn_affine))
                net.add(act_fun)
                
                if need_lin_comb:
                    temp = nn.Sequential()
                    temp.add(nn.Conv2d(num_channels, num_channels, 1, 1, padding=0, bias=bias))
                    temp.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
                    temp.add(act_fun)
                    net.add(temp)
                
            nc = num_channels
            if need_last:
                temp = nn.Sequential()
                temp.add( nn.Conv2d(nc, decoders_last_channels[n], 1, 1, padding=0, bias=bias) )
                temp.add(nn.BatchNorm2d( decoders_last_channels[n], affine=bn_affine))
                temp.add(act_fun)
                net.add(temp)
                nc = decoders_last_channels[n]
            net.add(nn.Conv2d(nc, decoders_last_channels[n], 1, 1, padding=0, bias=bias))
            if self.sig is not None:
                net.add(self.sig)
            
            nets.append(net)
            del(net)
        
        self.net1 = nets[0]
        self.net2 = nets[1]
        self.net3 = nets[2]
        
        net4 = nn.Sequential()
        nc = sum(decoders_last_channels)
        if need_last:
            net4.add(nn.Conv2d(nc,num_output_channels,1,1,padding=0,bias=bias))
            net4.add(act_fun)
            net4.add(nn.BatchNorm2d( num_output_channels, affine=bn_affine))
            nc = num_output_channels
        net4.add(nn.Conv2d(nc,num_output_channels,1,1,padding=0,bias=bias))
        self.net4 = net4
        
    def forward(self,x,scale_out=1):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        
        last_inp = torch.cat([out1,out2,out3],1)
        out = self.net4(last_inp)
        if self.sig is not None:
            out = self.sig(out)
        return out*scale_out
def parcdecoder(out_size = [256,256],
        in_size = [16,16],
        num_output_channels=3, 
        num_channels=128,
        decoders_numlayers_list = [2,4,6], # (ascending order) determines the number of layers per each decoder in the parallel structure
        decoders_last_channels = [20,20,20], # last layer channel contribution of each decoder
        need_sigmoid=True,
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_affine = True,
        nonlin_scales=False,
        bias=True,
        kernel_size=[3]*3,
        need_lin_comb=True,
        need_last=True,
        ):
    
    hidden_size = []
    for num_layers in decoders_numlayers_list:
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
        if nonlin_scales:
            xscales = np.ceil( np.linspace(scale_x * in_size[0],out_size[0],num_layers-1) )
            yscales = np.ceil( np.linspace(scale_y * in_size[1],out_size[1],num_layers-1) )
            h_s = [(int(x),int(y)) for (x,y) in zip(xscales,yscales)]
        else:
            h_s = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
        hidden_size.append(h_s)
    print(hidden_size)
    
    if need_sigmoid:
        sig = nn.Sigmoid()
    else:
        sig = None
    
    model = catc_model(decoders_numlayers_list,
                      decoders_last_channels,
                      num_channels, 
                      num_output_channels,
                      upsample_mode,
                      act_fun,
                      hidden_size,
                      sig = sig,
                      bn_affine = bn_affine,
                      bias=bias,
                      kernel_size=kernel_size,
                      need_lin_comb=need_lin_comb,
                      need_last=need_last
                      )
   
    return model