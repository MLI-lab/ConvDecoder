import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy



dtype = torch.FloatTensor  # This code is meant for CPU

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv1(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv1d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)



# Define the upsampling matrices
def get_upsample_matrix(k, identity=False, upsample_mode='linear'):
    # Returns a 2*k-1 x k numpy array corresponding to an upsampling matrix

    if identity:
        return np.eye(k)
    U = np.zeros((2*k-1, k))
    for i in range(k):
        U[2*i, i] = 1

        if i < k-1:
            if upsample_mode=='linear':
                U[2*i+1, [i, (i+1) % k]] = [1./2, 1./2]
            elif upsample_mode=='convex0.7-0.3':
                U[2*i+1, [i, (i+1) % k]] = [0.7, 0.3]
            elif upsample_mode=='convex0.75-0.25':
                U[2*i+1, [i, (i+1) % k]] = [0.75, 0.25]
    return U



class Upsample_Module(nn.Module):
    # Only works for batch size 1.  Works for any number of channels

    def __init__(self, upsample_mode='linear'):
        super(Upsample_Module,self).__init__()
        self.upsample_mode=upsample_mode
        
    def forward(self, x):
        n = x.shape[2]
        U = Variable(torch.Tensor(get_upsample_matrix(n, upsample_mode=self.upsample_mode)))
        return torch.stack([torch.t(U.matmul(torch.t(x[0,...])))], 0)




def decoder_1d(
        num_output_channels=1, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=False, 
        pad='zero', 
        upsample_mode='linear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        need_bn=True,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    #print('n_scales = %d' %n_scales)
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    model = nn.Sequential()

    for i in range(len(num_channels_up)-1):

        if upsample_mode!='none' and i!=0:
            if upsample_mode=='MatrixUpsample':
                model.add(Upsample_Module())
            elif upsample_mode=='MatrixUpsampleConvex0.7-0.3':
                model.add(Upsample_Module(upsample_mode='convex0.7-0.3'))
            elif upsample_mode=='MatrixUpsampleConvex0.75-0.25':
                model.add(Upsample_Module(upsample_mode='convex0.75-0.25'))
            elif upsample_mode=='nnUpsampleDouble':
                model.add(nn.Upsample(scale_factor=2.0, mode='linear', align_corners=False))
            elif upsample_mode=='nearest':
                model.add(nn.Upsample(scale_factor=2.0, mode='nearest'))

        model.add(conv1( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))
        if i != len(num_channels_up)-1:	
            if need_bn:
                model.add(nn.BatchNorm1d( num_channels_up[i+1] ))
            model.add(act_fun)
    
    model.add(conv1( num_channels_up[-1], num_output_channels, 1, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model


def fit_1d(net, 
	img_noisy_var, 
	num_channels, 
	img_clean_var,       
	net_input,           # Passing in the net_input is required
	num_iter = 5000, 
	LR = 0.01, 
	OPTIMIZER='adam', 
	opt_input = False, 
	reg_noise_std = 0, 
	reg_noise_decayevery = 100000, 
	mask_var = None, 
	apply_f = None, 
	decaylr = False, 
	net_input_gen = "random", 
        plot_output_every = None,
        ): 
                     
    net_input_saved = net_input.data.clone() 
    noise = net_input.data.clone() 
    p = [x for x in net.parameters() ] 
 
    if(opt_input == True): 
        net_input.requires_grad = True 
        p += [net_input] 
 
    mse_wrt_noisy = np.zeros(num_iter) 
    mse_wrt_truth = np.zeros(num_iter) 
     
    if OPTIMIZER == 'SGD': 
        print("optimize with SGD", LR) 
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9) 
    elif OPTIMIZER == 'adam': 
        print("optimize with adam", LR) 
        optimizer = torch.optim.Adam(p, lr=LR) 
 
    mse = torch.nn.MSELoss() #.type(dtype) 
    noise_energy = mse(img_noisy_var, img_clean_var) 



    for i in range(num_iter):  
        if decaylr is True: 
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=100) 
        if reg_noise_std > 0: 
            if i % reg_noise_decayevery == 0: 
                reg_noise_std *= 0.7 
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std)) 
        optimizer.zero_grad() 
        out = net(net_input.type(dtype)) 

        
         
        # training loss 
        if mask_var is not None: 
            loss = mse( out * mask_var , img_noisy_var * mask_var ) 
        elif apply_f: 
            loss = mse( apply_f(out) , img_noisy_var ) 
        else: 
            loss = mse(out, img_noisy_var) 
        loss.backward() 
        mse_wrt_noisy[i] = loss.data.cpu().numpy() 
        if mse_wrt_noisy[i] == np.min(mse_wrt_noisy[:i+1]):
            best_net = copy.deepcopy(net)
            best_mse_wrt_noisy = mse_wrt_noisy[i]
         
        # the actual loss 
        true_loss = mse(Variable(out.data, requires_grad=False), img_clean_var) 
        mse_wrt_truth[i] = true_loss.data.cpu().numpy() 
        if i % 10 == 0: 
            out2 = net(Variable(net_input_saved).type(dtype)) 
            loss2 = mse(out2, img_clean_var) 
            print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f' 
                   % (i, loss.data.item(),true_loss.data.item(),loss2.data.item(),noise_energy.data.item()), '\r', end='') 
        if plot_output_every and (i % plot_output_every==1):
            out3 = net(Variable(net_input_saved).type(dtype)) 
            ax = plt.figure(figsize=(12,5))
            plt.plot(out3[0,0,:].data.numpy(), '.b')
            plt.plot(img_clean_var[0,0,:].data.numpy(), '-r')
            plt.show()
        optimizer.step() 
    return mse_wrt_noisy, mse_wrt_truth,net_input_saved, best_net, best_mse_wrt_noisy  # Didn't implement case wehere there is noise in signal




