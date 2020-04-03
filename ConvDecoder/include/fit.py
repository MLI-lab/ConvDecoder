from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim

from .helpers import *
from .mri_helpers import *

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def sqnorm(a):
    return np.sum( a*a )

def get_distances(initial_maps,final_maps):
    results = []
    for a,b in zip(initial_maps,final_maps):
        res = sqnorm(a-b)/(sqnorm(a) + sqnorm(b))
        results += [res]
    return(results)

def get_weights(net):
    weights = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            weights += [m.weight.data.cpu().numpy()]
    return weights

def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        mask = None,
        apply_f = None,
        lr_decay_epoch = 0,
        net_input = None,
        net_input_gen = "random",
        lsimg=None,
        find_best=False,
        weight_decay=0,
        upsample_mode = "bilinear",
        totalupsample = 1,
        loss_type="MSE",
        output_gradients=False,
        output_weights=False,
        show_images=False,
        plot_after=None,
        in_size=None,
       ):

    if net_input is not None:
        print("input provided")
    else:
        
        if upsample_mode=="bilinear":
            # feed uniform noise into the network 
            totalupsample = 2**len(num_channels)
            width = int(img_clean_var.data.shape[2]/totalupsample)
            height = int(img_clean_var.data.shape[3]/totalupsample)
        elif upsample_mode=="deconv":
            # feed uniform noise into the network 
            totalupsample = 2**(len(num_channels)-1)
            width = int(img_clean_var.data.shape[2]/totalupsample)
            height = int(img_clean_var.data.shape[3]/totalupsample)
        elif upsample_mode=="free":
            width,height = in_size
            
        
        shape = [1,num_channels[0], width, height]
        print("input shape: ", shape)
        net_input = Variable(torch.zeros(shape)).type(dtype)
        net_input.data.uniform_()
        net_input.data *= 1./10
    
    net_input = net_input.type(dtype)
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters() ]

    if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)
    
    
    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    elif OPTIMIZER == 'LBFGS':
        print("optimize with LBFGS", LR)
        optimizer = torch.optim.LBFGS(p, lr=LR)

    if loss_type=="MSE":
        mse = torch.nn.MSELoss() #.type(dtype) 
    if loss_type=="L1":
        mse = nn.L1Loss()
    
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    nconvnets = 0
    for p in list(filter(lambda p: len(p.data.shape)>2, net.parameters())):
        nconvnets += 1
    
    out_grads = np.zeros((nconvnets,num_iter))
        
    init_weights = get_weights(net)
    out_weights = np.zeros(( len(init_weights) ,num_iter))
    
    out_imgs = np.zeros((1,1))
    
    if plot_after is not None:
        out_img_np = net( net_input_saved.type(dtype) ).data.cpu().numpy()[0]
        out_imgs = np.zeros( (len(plot_after),) + out_img_np.shape )
    
    PSNR = []
    SSIM = []
    for i in range(num_iter):
        
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
        def closure():
            optimizer.zero_grad()
            out = net(net_input.type(dtype))

            # training loss 
            if mask_var is not None:
                loss = mse( out * mask_var , img_noisy_var * mask_var )
            elif apply_f:
                loss = mse( apply_f(out,mask) , img_noisy_var )
            else:
                loss = mse(out, img_noisy_var)
        
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            
            
            # the actual loss 
            true_loss = mse( Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype) )
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            if output_gradients:
                for ind,p in enumerate(list(filter(lambda p: p.grad is not None and len(p.data.shape)>2, net.parameters()))):
                    out_grads[ind,i] = p.grad.data.norm(2).item()
                    #print(p.grad.data.norm(2).item())
                    #su += p.grad.data.norm(2).item()
                    #mse_wrt_noisy[i] = su
            
            if i % 100 == 0:
                if lsimg is not None:
                    ### compute ssim and psnr ###
                    out_chs = out.data.cpu().numpy()[0]
                    out_imgs = channels2imgs(out_chs)
                    # least squares reconstruciton
                    orig = crop_center2( root_sum_of_squares2(var_to_np(lsimg)) , 320,320)

                    # deep decoder reconstruction
                    rec = crop_center2(root_sum_of_squares2(out_imgs),320,320)

                    ssim_const = ssim(orig,rec,data_range=orig.max())
                    SSIM.append(ssim_const)

                    psnr_const = psnr(orig,rec,np.max(orig))
                    PSNR.append(psnr_const)
                    ### ###
                out2 = net(Variable(net_input_saved).type(dtype))
                loss2 = mse(out2, img_clean_var)
                print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f' % (i, loss.data,true_loss.data,loss2.data), '\r', end='')
            
            if show_images:
                if i % 50 == 0:
                    print(i)
                    out_img_np = net( ni.type(dtype) ).data.cpu().numpy()[0]
                    myimgshow(plt,out_img_np)
                    plt.show()
                    
            if plot_after is not None:
                if i in plot_after:
                    out_imgs[ plot_after.index(i) ,:] = net( net_input_saved.type(dtype) ).data.cpu().numpy()[0]
            
            if output_weights:
                out_weights[:,i] = np.array( get_distances( init_weights, get_weights(net) ) )
            
            return loss   
        
        loss = optimizer.step(closure)
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(net)
                if opt_input:
                    best_ni = net_input.data.clone()
                else:
                    best_ni = net_input_saved.clone()
                 
        
    if find_best:
        net = best_net
        net_input_saved = best_ni
    if output_gradients and output_weights:
        return SSIM,PSNR,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_grads
    elif output_gradients:
        return SSIM,PSNR,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_grads      
    elif output_weights:
        return SSIM,PSNR,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_weights
    elif plot_after is not None:
        return SSIM,PSNR,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_imgs
    else:
        return SSIM,PSNR,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net       
        





        ### weight regularization
        #if orth_reg > 0:
        #    for name, param in net.named_parameters():
                # consider all the conv weights, but the last one which only combines colors
        #        if '.1.weight' in name and str( len(net)-1 ) not in name:
        #            param_flat = param.view(param.shape[0], -1)
        #            sym = torch.mm(param_flat, torch.t(param_flat))
        #            sym -= Variable(torch.eye(param_flat.shape[0])).type(dtype)
        #            loss = loss + (orth_reg * sym.sum().type(dtype) )
        ###
        
def fit_multiple(net,
        imgs, # list of images [ [1, color channels, W, H] ] 
        num_channels,
        num_iter = 5000,
        LR = 0.01,
        find_best=False,
        upsample_mode="bilinear",
       ):
    # generate netinputs
    # feed uniform noise into the network
    nis = []
    for i in range(len(imgs)):
        if upsample_mode=="bilinear":
            # feed uniform noise into the network 
            totalupsample = 2**len(num_channels)
        elif upsample_mode=="deconv":
            # feed uniform noise into the network 
            totalupsample = 2**(len(num_channels)-1)
            #totalupsample = 2**len(num_channels)
        width = int(imgs[0].data.shape[2]/totalupsample)
        height = int(imgs[0].data.shape[3]/totalupsample)
        shape = [1 ,num_channels[0], width, height]
        print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10
        nis.append(net_input)

    # learnable parameters are the weights
    p = [x for x in net.parameters() ]

    mse_wrt_noisy = np.zeros(num_iter)

    optimizer = torch.optim.Adam(p, lr=LR)

    mse = torch.nn.MSELoss() #.type(dtype) 

    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    for i in range(num_iter):
        
        def closure():
            optimizer.zero_grad()
            
            #loss = np_to_var(np.array([0.0]))
            out = net(nis[0].type(dtype))
            loss = mse(out, imgs[0].type(dtype)) 
            #for img,ni in zip(imgs,nis):
            for j in range(1,len(imgs)):
                #out = net(ni.type(dtype))
                #loss += mse(out, img.type(dtype))
                out = net(nis[j].type(dtype))
                loss += mse(out, imgs[j].type(dtype))
        
            #out = net(nis[0].type(dtype))
            #out2 = net(nis[1].type(dtype))
            #loss = mse(out, imgs[0].type(dtype)) + mse(out2, imgs[1].type(dtype))
        
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            
            if i % 10 == 0:
                print ('Iteration %05d    Train loss %f' % (i, loss.data), '\r', end='')
            return loss
        
        loss = optimizer.step(closure)
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(net)
                       
    if find_best:
        net = best_net
    return mse_wrt_noisy, nis, net        
        
      