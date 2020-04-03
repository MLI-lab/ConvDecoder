import torch 
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

from torch.autograd import Variable

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import PIL

from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

from . import transforms as transform
from .helpers import var_to_np,np_to_var

def ksp2measurement(ksp):
    return np_to_var( np.transpose( np.array([np.real(ksp),np.imag(ksp)]) , (1, 2, 3, 0)) )   

def lsreconstruction(measurement,mode='both'):
    # measurement has dimension (1, num_slices, x, y, 2)
    fimg = transform.ifft2(measurement)
    normimag = torch.norm(fimg[:,:,:,:,0])
    normreal = torch.norm(fimg[:,:,:,:,1])
    #print("real/img parts: ",normimag, normreal)
    if mode == 'both':
        return torch.sqrt(fimg[:,:,:,:,0]**2 + fimg[:,:,:,:,1]**2)
    elif mode == 'real':
        return torch.tensor(fimg[:,:,:,:,0]) #torch.sqrt(fimg[:,:,:,:,0]**2)
    elif mode == 'imag':
        return torch.sqrt(fimg[:,:,:,:,1]**2)

def root_sum_of_squares2(lsimg):
    out = np.zeros(lsimg[0].shape)
    for img in lsimg:
        out += img**2
    return np.sqrt(out)

def crop_center2(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def channels2imgs(out):
    sh = out.shape
    chs = int(sh[0]/2)
    imgs = np.zeros( (chs,sh[1],sh[2]) )
    for i in range(chs):
        imgs[i] = np.sqrt( out[2*i]**2 + out[2*i+1]**2 )
    return imgs

def forwardm(img,mask):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)
    mask = np_to_var(mask)[0].type(dtype)
    s = img.shape
    ns = int(s[1]/2) # number of slices
    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
    for i in range(ns):
        fimg[0,i,:,:,0] = img[0,2*i,:,:]
        fimg[0,i,:,:,1] = img[0,2*i+1,:,:]
    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
    for i in range(ns):
        Fimg[0,i,:,:,0] *= mask
        Fimg[0,i,:,:,1] *= mask
    return Fimg

def get_scale_factor(net,num_channels,in_size,slice_ksp): #,mask,threshold = 0.02,search_range=np.linspace(0,20,100)):
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    shape = [1,num_channels, in_size[0], in_size[1]]
    ni = Variable(torch.zeros(shape)).type(dtype)
    ni.data.uniform_()
    # generate random image
    out_chs = net( ni.type(dtype) ).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    out_img_tt = transform.root_sum_of_squares( torch.tensor(out_imgs) , dim=0)

    ### get norm of least-squares reconstruction
    ksp_tt = transform.to_tensor(slice_ksp)
    orig_tt = transform.ifft2(ksp_tt)           # Apply Inverse Fourier Transform to get the complex image
    orig_imgs_tt = transform.complex_abs(orig_tt)   # Compute absolute value to get a real image
    orig_img_tt = transform.root_sum_of_squares(orig_imgs_tt, dim=0)
    orig_img_np = orig_img_tt.cpu().numpy()

    s = np.linalg.norm(out_img_tt) / np.linalg.norm(orig_img_np)
    
    """
    slice_ksp_torchtensor = transform.to_tensor(slice_ksp)      # Convert from numpy array to pytorch tensor
    slice_image = transform.ifft2(slice_ksp_torchtensor)           # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = transform.complex_abs(slice_image)   # Compute absolute value to get a real image
    
    best_diff = 1
    best_s = 1
    best_norms = (0,0)
    for s in search_range:

        scale_factor = 1./( np.max( slice_image_abs.cpu().numpy() ) / s ) # 3.1 works well

        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor*scale_factor, mask = mask)
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        
        # get net input
        shape = [1,num_channels, in_size[0], in_size[1]]
        ni = Variable(torch.zeros(shape)).type(dtype)
        ni.data.uniform_()

        # generate random image
        out_chs = net( ni.type(dtype) ).data.cpu().numpy()[0]
        out_imgs = channels2imgs(out_chs)

        rec = root_sum_of_squares2(out_imgs)

        # least squares reconstruciton
        orig = root_sum_of_squares2(var_to_np(lsreconstruction(unders_measurement))) 
        
        orig_norm = np.linalg.norm(orig)
        rec_norm = np.linalg.norm(rec)
        diff = abs( orig_norm - rec_norm ) / min(orig_norm,rec_norm)
        
        if diff < threshold:
            print("{} relative norm difference found.".format(round(diff,3)))
            print("2 norms: {}, {}".format(orig_norm,rec_norm))
            return s
        if diff < best_diff:
            best_diff = diff
            best_s = s
            best_norms = (orig_norm,rec_norm)
    print("best scaling factor found gives {} relative norm difference.".format(round(best_diff,3)))
    print("2 norms: {}, {}".format(best_norms[0],best_norms[1]))
    return best_s
    """
    return s