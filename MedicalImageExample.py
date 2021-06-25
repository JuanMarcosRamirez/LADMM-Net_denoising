# Generate noisy image data set
# Juan Marcos Ramirez Rondon, Universidad Rey Juan Carlos, Spain

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from LadmmNet import LADMMgrayscaleNet
import torch
import torch.nn as nn

gpu_list = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='LADMM-Net')
parser.add_argument('--epochs', type=int, default=128, help='number of epochs')
parser.add_argument('--layer_num', type=int, default=12, help='number of ISTA-Net layers')
parser.add_argument('--SNR', type=float, default=20, help='batch size')
parser.add_argument('--noise_type', type=str, default='poisson', help='Gaussian noise (gaussian) or Poisson noise (poisson)')

args = parser.parse_args()

layer_num       = args.layer_num
epochs          = args.epochs
SNR             = args.SNR
noise_type      = args.noise_type

M = 256
N = 256

model   = LADMMgrayscaleNet(layer_num)
model   = nn.DataParallel(model)
model   = model.to(device)

# Loading network parameters
model_dir = "parameters/Poisson/LADMM_Net_layer_%d_SNR_%ddB" % (layer_num, SNR)
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epochs)))


# Loading noisy image
# If you need to remove noise to a 256 x 256 medical image, include the image 
# in a .mat format in the medical_image folder. Change the file name.
noisy_image = sio.loadmat('medical_image/noisy_slice.mat')['In'] 
y_np        = noisy_image.reshape((M*N),order='F') 
y       = torch.from_numpy(y_np).type(torch.FloatTensor) * (1/255.0)
y       = torch.t(y).to(device)
del y_np

[x_output, loss_layers_sym] = model(y, M, N)
x_output = x_output.cpu().detach().numpy() * 255.0
x_output = np.transpose(np.reshape(x_output, (M,N)))

plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.title('Noisy')
plt.imshow(noisy_image, cmap='gray')
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.title('Denoised')
plt.imshow(x_output, cmap='gray')