import torch
import torch.nn as nn

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

# Residual block
class ParB(nn.Module):
    def __init__(self, decoders_channels,decoders_last_channels,num_channels, num_output_channels, upsample_mode,act_fun,bn_af,sig=None):
        super(ParB, self).__init__()
        
        self.upsamp = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.act_fun = act_fun
        self.sig = sig
        #self.relu = nn.ReLU()
        #self.sig= nn.Sigmoid()
        self.batchN = nn.BatchNorm2d( num_channels, affine=bn_af)
        
        self.convs = []
        for c,dc in enumerate(decoders_channels):
            decoder_convs = [nn.Conv2d(num_channels, num_channels, 1, 1, padding=0, bias=False) for i in range(dc)]
            decoder_convs.append(nn.Conv2d(num_channels, decoders_last_channels[c], 1, 1, padding=0, bias=False))
            self.convs.append(decoder_convs)
        print(len(self.convs),self.convs[0])
        self.last_conv = nn.Conv2d(sum(decoders_last_channels), num_output_channels, 1, 1, padding=0, bias=False)
        """
        self.conv10 = nn.Conv2d(num_channels_up[0], num_channels_up[1], 1, 1, padding=0, bias=False)
        self.conv11 = nn.Conv2d(num_channels_up[1], 30, 1, 1, padding=0, bias=False)
        
        self.conv20 = nn.Conv2d(num_channels_up[0], num_channels_up[1], 1, 1, padding=0, bias=False)
        self.conv21 = nn.Conv2d(num_channels_up[1], num_channels_up[2], 1, 1, padding=0, bias=False)
        self.conv22 = nn.Conv2d(num_channels_up[2], num_channels_up[3], 1, 1, padding=0, bias=False)
        self.conv23 = nn.Conv2d(num_channels_up[3], 20, 1, 1, padding=0, bias=False)
        
        self.conv30 = nn.Conv2d(num_channels_up[0], num_channels_up[1], 1, 1, padding=0, bias=False)
        self.conv31 = nn.Conv2d(num_channels_up[1], num_channels_up[2], 1, 1, padding=0, bias=False)
        self.conv32 = nn.Conv2d(num_channels_up[2], num_channels_up[3], 1, 1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(num_channels_up[3], num_channels_up[4], 1, 1, padding=0, bias=False)
        self.conv34 = nn.Conv2d(num_channels_up[4], num_channels_up[5], 1, 1, padding=0, bias=False) 
        self.conv35 = nn.Conv2d(num_channels_up[5], 10, 1, 1, padding=0, bias=False)
        
        self.conv = nn.Conv2d(60, num_output_channels, 1, 1, padding=0, bias=False)
        """
    def forward(self, x):
        outs = []
        for i in range(len(self.convs)):
            out = self.convs[i][0](x)
            out = self.batchN(self.act_fun(self.upsamp(out)))
            for cnv in self.convs[i][1:]:
                out = cnv(x)
                out = self.batchN(self.act_fun(self.upsamp(out)))
            outs.append(out)
        """
        out1 = self.conv10(x)
        out1 = self.batchN(self.relu(self.upsamp(out1)))
        out1 = self.conv11(out1)
        out1 = self.sig(self.upsamp(self.upsamp(self.upsamp(out1))))
        
        out2 = self.conv20(x)
        out2 = self.batchN(self.relu(self.upsamp(out2)))
        out2 = self.conv21(out2)
        out2 = self.batchN(self.relu(self.upsamp(out2)))
        out2 = self.conv22(out2)
        out2 = self.batchN(self.relu(self.upsamp(out2))) 
        out2 = self.conv23(out2)
        out2 = self.sig(self.upsamp(out2))
        
        out3 = self.conv30(x)
        out3 = self.batchN(self.relu(self.upsamp(out3)))
        out3 = self.conv31(out3)
        out3 = self.batchN(self.relu(self.upsamp(out3)))
        out3 = self.conv32(out3)
        out3 = self.batchN(self.relu(self.upsamp(out3)))
        out3 = self.conv33(out3)
        out3 = self.batchN(self.relu(self.upsamp(out3)))
        out3 = self.conv34(out3)
        out3 = self.batchN(self.relu(out3)) 
        out3 = self.conv35(out3)
        out3 = self.sig(out3)
        """
        last_in = torch.cat(outs,1)
        out = self.last_conv(last_in)
        if self.sig is not None:
            out = self.sig(out)
        return out

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
        #print(out1)
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
    
    #num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    #n_scales = len(num_channels_up) 
    if need_sigmoid:
        sig = nn.Sigmoid()
    model = cat_model(decoders_numlayers_list,
                      decoders_last_channels,
                      num_channels, 
                      num_output_channels,
                      upsample_mode,
                      act_fun,
                      sig = sig,
                      bn_affine = bn_affine,
                      pad = pad)
    """
    
    model = nn.Sequential()
    model.add(ParB(decoders_channel_list,decoders_last_channels,num_channels, num_output_channels, upsample_mode,act_fun,sig,bn_affine))
    
    nets = []
    M = max(decoders_numlayers_list)
    for n,num_layers in enumerate(decoders_numlayers_list):
        nc = num_channels
        net = nn.Sequential()
        for i in range(num_layers):
            if i == num_layers-1:
                nc = decoders_last_channels[n]
            net.add(conv( num_channels, nc,  1, 1, pad=pad))
            net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            net.add(act_fun)
            net.add(nn.BatchNorm2d( nc, affine=bn_affine))
            if i == num_layers-1:
                if num_layers != M: # upsample the output to match the output size of all networks
                    for j in range(M-num_layers):
                        net.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        nets.append(net)
        del(net)
        
    #model = nn.ConcatTable(1)
    model = nn.Sequential(*(net for net in nets))
    #model.add(torch.cat(nets,1))
    model.add(conv(sum(decoders_last_channels),num_output_channels,1,pad=pad))
    
    for net in nets:
        model.add(net)
        final_model = nn.Sequential()
    final_model.add(model)
    final_model.add(conv(sum(decoders_last_channels),num-output_channels,1,pad=pad))
    
    if need_sigmoid:
        final_model.add(nn.Sigmoid())
    
    if need_sigmoid:
        model.add(nn.Sigmoid())
    """
    return model