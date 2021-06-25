# Generate noisy image data set
# Juan Marcos Ramirez Rondon, Universidad Rey Juan Carlos, Spain

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class LADMMgrayscaleBlock(torch.nn.Module):
    def __init__(self):
        super(LADMMgrayscaleBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.1]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.rh2_prmt = nn.Parameter(torch.Tensor([0.1]))          

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1,  3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        
    def forward(self, x, d, r, y, M, N):
        x = x - self.lambda_step * x
        x = x - self.lambda_step * (self.rh2_prmt * r)
        x_upd = x + self.lambda_step * y
        
        # Forward transform block                
        x_input = x_upd.view(-1, 1, M, N)
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        
        
        # Soft thresholding unit
        x = x_forward + d
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.soft_thr))
        
        r = r.view(-1, 1, M, N)
        d = d + x_forward - x
        r = x_forward + d - x

        # Inverse transform block
        r = F.conv2d(r, self.conv1_backward, padding=1)
        r = F.relu(r)
        r = F.conv2d(r, self.conv2_backward, padding=1) 
        r = r.view(-1,M*N)
        
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        
        symloss = x_est - x_input
        # x_est = x_est.view(-1,M*N)
        return [x_upd, d, r, symloss]


class LADMMgrayscaleNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(LADMMgrayscaleNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            onelayer.append(LADMMgrayscaleBlock())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, y, M, N):
        x = y
        r = 0.00 * x
        d = torch.zeros((1,32,M,N)).cuda()
        layers_sym = []
        for i in range(self.LayerNo):
            [x, d, r, layer_sym] = self.fcs[i](x, d, r, y, M, N)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]