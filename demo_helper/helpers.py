import torch
import numpy as np
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

class MaskFunc:
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    MaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Converts image in numpy.array to torch.Variable.
    
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(torch.from_numpy(img_np)[None, :])

def var_to_np(img_var):
    '''
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

def ksp2measurement(ksp):
    return np_to_var( np.transpose( np.array([np.real(ksp),np.imag(ksp)]) , (1, 2, 3, 0)) )   

def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))

def crop_center(img,cropx,cropy):
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
    Fimg = fft2(fimg) # dim: (1,num_slices,x,y,2)
    for i in range(ns):
        Fimg[0,i,:,:,0] *= mask
        Fimg[0,i,:,:,1] *= mask
    return Fimg

def get_mask(slice_ksp_torchtensor, slice_ksp,factor=4,cent=0.07):
    try: # if the file already has a mask
        temp = np.array([1 if e else 0 for e in f["mask"]])
        temp = temp[np.newaxis].T
        temp = np.array([[temp]])
        mask = to_tensor(temp).type(dtype).detach().cpu()
    except: # if we need to create a mask
        desired_factor = factor # desired under-sampling factor
        undersampling_factor = 0
        tolerance = 0.03
        while undersampling_factor < desired_factor - tolerance or undersampling_factor > desired_factor + tolerance:
            mask_func = MaskFunc(center_fractions=[cent], accelerations=[desired_factor])  # Create the mask function object
            masked_kspace, mask = apply_mask(slice_ksp_torchtensor, mask_func=mask_func)   # Apply the mask to k-space
            mask1d = var_to_np(mask)[0,:,0]
            undersampling_factor = len(mask1d) / sum(mask1d)

    mask1d = var_to_np(mask)[0,:,0]

    # The provided mask and data have last dim of 368, but the actual data is smaller.
    # To prevent forcing the network to learn outside the data region, we force the mask to 0 there.
    mask1d[:mask1d.shape[-1]//2-160] = 0 
    mask1d[mask1d.shape[-1]//2+160:] =0
    mask2d = np.repeat(mask1d[None,:], slice_ksp.shape[1], axis=0).astype(int) # Turning 1D Mask into 2D that matches data dimensions
    mask2d = np.pad(mask2d,((0,),((slice_ksp.shape[-1]-mask2d.shape[-1])//2,)),mode='constant') # Zero padding to make sure dimensions match up
    mask = to_tensor( np.array( [[mask2d[0][np.newaxis].T]] ) ).type(dtype).detach().cpu()
    return mask, mask1d, mask2d

def apply_mask(data, mask_func = None, mask = None, seed=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    if mask is None:
        mask = mask_func(shape, seed)
    return data * mask, mask

def fft(input, signal_ndim, normalized=False):
  # This function is called from the fft2 function below
  if signal_ndim < 1 or signal_ndim > 3:
    print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
    return

  dims = (-1)
  if signal_ndim == 2:
    dims = (-2, -1)
  if signal_ndim == 3:
    dims = (-3, -2, -1)

  norm = "backward"
  if normalized:
    norm = "ortho"

  return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=dims, norm=norm))

def ifft(input, signal_ndim, normalized=False):
  # This function is called from the ifft2 function below
  if signal_ndim < 1 or signal_ndim > 3:
    print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
    return

  dims = (-1)
  if signal_ndim == 2:
    dims = (-2, -1)
  if signal_ndim == 3:
    dims = (-3, -2, -1)

  norm = "backward"
  if normalized:
    norm = "ortho"

  return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=dims, norm=norm))

def fft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_abs(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

def fftshift(x, dim=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def roll(x, shift, dim):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
