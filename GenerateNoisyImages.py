# Generate noisy image data set
# Juan Marcos Ramirez Rondon, Universidad Rey Juan Carlos, Spain

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import misc

parser = ArgumentParser(description='noisy_gray_images')

parser.add_argument('--SNR', type=float, default=20, help='signal-to-noise ratio in dB')
parser.add_argument('--image_set', type=str, default='test_set', help='training set (train_set) or test set (test_set)')
parser.add_argument('--noise_type', type=str, default='poisson', help='Gaussian noise (gaussian) or Poisson noise (poisson)')

args = parser.parse_args()

SNR         = args.SNR
image_set   = args.image_set
noise_type  = args.noise_type

if (image_set == 'train_set'):
    for i in range(0,80):
        input_file  = 'data/train_images/gray_%03d.mat'%(i+1)
        Ii          = sio.loadmat(input_file)['I'] * 1.0
        M, N        = np.shape(Ii)
        if (noise_type == 'gaussian'):
            output_file = 'data/train_images/Gaussian_SNR_%ddB/noisy_%03d.mat'%(SNR,i+1)
            sigma       = np.sqrt(np.mean(np.power(Ii,2))/np.power(10,(SNR/10)))
            Io          = Ii + np.random.normal(0,sigma,(M,N))
        if (noise_type == 'poisson'):
            output_file = 'data/train_images/Poisson_SNR_%ddB/noisy_%03d.mat'%(SNR,i+1)
            Io           = np.multiply(Ii / np.power(10,SNR/10),np.random.poisson(np.power(10,SNR/10),(M,N)))
        sio.savemat(output_file,{"In":Io})
if (image_set == 'test_set'):
    for i in range(0,12):
        input_file  = 'data/test_images/gray_%03d.mat'%(i+1)
        Ii          = sio.loadmat(input_file)['I'] * 1.0
        M, N        = np.shape(Ii)
        if (noise_type == 'gaussian'):
            output_file = 'data/test_images/Gaussian_SNR_%ddB/noisy_%03d.mat'%(SNR,i+1)
            sigma       = np.sqrt(np.mean(np.power(Ii,2))/np.power(10,(SNR/10)))
            Io          = Ii + np.random.normal(0,sigma,(M,N))                
        if (noise_type == 'poisson'):
            output_file = 'data/test_images/Poisson_SNR_%ddB/noisy_%03d.mat'%(SNR,i+1)
            Io           = np.multiply(Ii / np.power(10,SNR/10),np.random.poisson(np.power(10,SNR/10),(M,N)))  
        sio.savemat(output_file,{"In":Io})
        # SNR_output  = 10 * np.log10(np.mean(np.power(Ii,2)) / np.mean(np.power((Ii-(1.0*Io)),2)))
        # print(SNR_output)