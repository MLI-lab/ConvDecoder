import torch
import torch.nn as nn

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class cat_model(nn.Module):
    def __init__(self, decoders_numlayers_list,decoders_last_channels,num_channels, num_output_channels,
                 upsample_mode,act_fun,sig=None,bn_affine=True,pad=0):
        super(cat_model, self).__init__()
        nets = []
        M = max(decoders_numlayers_list)
        for n,num_layers in enumerate(decoders_numlayers_list):
            nc = num_channels
            net = nn.Sequential()
            for i in range(num_layers-1):
                net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                net.add(nn.Conv2d(num_channels, nc, 1, 1, padding=0, bias=False))
                net.add(act_fun)
                net.add(nn.BatchNorm2d( nc, affine=bn_affine))
                    
            nc = decoders_last_channels[n]
            net.add(nn.Conv2d(num_channels, nc, 1, 1, padding=0, bias=False))
            net.add(nn.Sigmoid())

            if num_layers != M: # upsample the output to match the output size of all networks
                for j in range(M-num_layers):
                    net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            nets.append(net)
            del(net)
        
        self.net1 = nets[0]
        self.net2 = nets[1]
        self.net3 = nets[2]
        self.last_conv = nn.Conv2d(sum(decoders_last_channels),num_output_channels,1,1,padding=0,bias=False)
        self.sig = sig

    def forward(self,x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        
        last_inp = torch.cat([out1,out2,out3],1)
        out = self.last_conv(last_inp)
        if self.sig is not None:
            out = self.sig(out)
        return out
def pardecoder(
        num_output_channels=3, 
        num_channels=128,
        decoders_numlayers_list = [2,4,6], # (ascending order) determines the number of layers per each decoder in the parallel structure
        decoders_last_channels = [20,20,20], # last layer channel contribution of each decoder
        need_sigmoid=True, 
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_affine = True,
        ):
     
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
                      sig = sig,
                      bn_affine = bn_affine,
                      pad = pad)
   
    return model