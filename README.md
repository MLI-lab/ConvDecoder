# ConvDecoder

This repository provides code for reproducing the results in the paper:

***

Code by: Mohammad Zalbagi Darestani (mz35@rice.edu) and Reinhard Heckel (rh43@rice.edu)

*****

The aim of the code is to investigate the capability of different un-trained methods, including our proposed ConvDecoder, for the MRI acceleration problem. The task is to recover a fine image from a few measurements. In this regard, we specifically provide experiments to: (i) compare ConvDecoder with U-net, a standard popular trained method for medical imaging, on the FastMRI validation set (ii) compare ConvDecoder with Deep Decoder and Deep Image Prior, two popular un-trained methods for standard inverse problems, again, on the FastMRI dataset, (iii) compare ConvDecoder with U-net on an out-of-distribution sample to demonstrate the robustness of un-trained methods toward a shift in the distribution at the inference time, and finally (iv) visualize the output of ConvDecoder layers to illustrate how ConvDecoder, as a convolutional generator, finds a fine representation of an image.

### List of contents

# Setup and installation
On a normal computer, it takes aproximately 5 minutes to install all the required softwares and packages.

### OS requirements
The code has been tested on the following operating system:

	Linux: Ubuntu 16.04.5
