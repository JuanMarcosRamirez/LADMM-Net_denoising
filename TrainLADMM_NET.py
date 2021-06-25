# Generate noisy image data set
# Juan Marcos Ramirez Rondon, Universidad Rey Juan Carlos, Spain

# Import libraries
import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io as sio
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
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--SNR', type=float, default=20, help='batch size')
parser.add_argument('--noise_type', type=str, default='poisson', help='Gaussian noise (gaussian) or Poisson noise (poisson)')

args = parser.parse_args()

layer_num       = args.layer_num
epochs          = args.epochs
learning_rate   = args.learning_rate
batch_size      = args.batch_size
SNR             = args.SNR
noise_type      = args.noise_type

train_samples   = 80

model   = LADMMgrayscaleNet(layer_num)
model   = nn.DataParallel(model)
model   = model.to(device)
print_flag = 0
if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if (noise_type == 'gaussian'):
    model_dir = "parameters/Gaussian/LADMM_Net_layer_%d_SNR_%ddB" % (layer_num, SNR)
if (noise_type == 'poisson'):
    model_dir = "parameters/Poisson/LADMM_Net_layer_%d_SNR_%ddB" % (layer_num, SNR)
    
# Routine parameters
M = 256
N = 256

psnr = np.zeros(epochs+1)
for epoch_i in range(0, epochs+1): 
    print('Epoch: %d'%(epoch_i))
    psnr_batch  = np.zeros(train_samples//batch_size)
    for j in range(0, train_samples//batch_size):
        y_np = np.zeros((M*N,batch_size))
        for k in range(0, batch_size):
            fpointer = j*batch_size + k + 1
            if (noise_type == 'gaussian'):
                In_file     = 'data/train_images/Gaussian_SNR_%ddB/noisy_%03d.mat'%(SNR,fpointer)
            if (noise_type == 'poisson'):
                In_file     = 'data/train_images/Poisson_SNR_%ddB/noisy_%03d.mat'%(SNR,fpointer)
            noisy_image = sio.loadmat(In_file)['In']
            y_np[:,k] = noisy_image.reshape((M*N),order='F')
            
        # Preparing input data
        y       = torch.from_numpy(y_np).type(torch.FloatTensor) * (1/255.0)
        y       = torch.t(y).to(device)
        del y_np, noisy_image        
        
        # Implementing the LADMM-Net
        [x_output, loss_layers_sym] = model(y, M, N)

        # Computing the loss function
        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1): 
            loss_constraint += float(torch.mean(torch.pow(loss_layers_sym[k+1], 2)))
            

        # Loading training outputs
        xv = np.zeros((M*N,batch_size))
        for k in range(0, batch_size):
            fpointer = j*batch_size + k + 1
            input_file  = 'data/train_images/gray_%03d.mat'%(fpointer)
            I        = sio.loadmat(input_file)['I']
            xv[:,k] = I.reshape((M*N),order='F')

        x   = torch.from_numpy(xv).type(torch.FloatTensor) * (1/255.0)
        x   = torch.t(x).to(device)
        del xv
        
        loss_discrepancy    = torch.mean(torch.pow(x_output - x, 2))         
        loss_all            = loss_discrepancy + torch.mul(0.01, loss_constraint)
        psnr_batch[j]       = torch.mul(10, torch.log10(torch.div(1.0, loss_discrepancy))) 

        # Network update
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()         

    # Computing the average PSNR 
    psnr[epoch_i] = np.mean(psnr_batch)
    sio.savemat('learning_curves/psnr_versus_epochs_layer_%d_SNR_%ddB.mat'%(layer_num, SNR) ,{"psnr":psnr})    
    print('PSNR = %.2f dB'%(psnr[epoch_i]))
    if epoch_i % 32 == 0:
        torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('PSNR [dB]')
plt.plot(psnr)                            