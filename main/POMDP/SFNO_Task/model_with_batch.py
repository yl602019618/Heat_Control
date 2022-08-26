import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
import torchvision.transforms.functional as vtF
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SuperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,resolution):
        super(SuperConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.resolution = resolution
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.resolution , self.resolution//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, self.modes1:2*self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights3)
        out_ft[:, :, -2*self.modes1:-self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights4)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.resolution, self.resolution))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,modes3,modes4,width,resolution,step):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.resolution = resolution
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        
        self.step = step # 
        self.last_layer = 128
        self.fc0 = nn.Linear(self.step, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SuperConv2d(self.width, self.width, self.modes1, self.modes2,self.resolution)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        #self.ln1 = nn.LayerNorm([self.resolution,self.resolution,self.last_layer])
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.fc1 = nn.Linear(self.width, self.last_layer)
        
        self.fc2 = nn.Linear(self.last_layer, 1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = self.bn0(x)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x0 = vtF.resize(x, size=[self.resolution,],interpolation=2)
        x2 = self.w0(x0)
        x = x1 + x2
        x = F.gelu(x)
        

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.bn1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



class Heat_forward(nn.Module):
    def __init__(self, n_x,dt , alpha,device, name=''):
        super(Heat_forward, self).__init__()
        '''
        operates on last 2 dimension
        '''


        self.n_x = n_x  
        self.dt = dt
        self.dx = 1/n_x
        self.alpha = alpha
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.r = self.alpha*self.dt/self.dx/self.dx
        self.device = device
        
        node = self.n_x*self.n_x
        d1 = np.ones((node,))*(-2*self.r+1)
        d2 = np.ones((node,))*(2*self.r+1)
        d3 = np.ones((node - 1,))*(self.r/2)
        d4 = np.ones((node - self.n_x,))*(self.r/2)
        A1 = torch.Tensor( np.diag(d2) - np.diag(d3,-1) - np.diag(d3,1) - np.diag(d4,- self.n_x) - np.diag(d4, self.n_x))
        A0 = torch.Tensor(np.diag(d1) + np.diag(d3,-1) + np.diag(d3,1) + np.diag(d4,- self.n_x) + np.diag(d4, self.n_x) )
        self.A1 = nn.Parameter(A1,requires_grad=False)
        self.A0 =  nn.Parameter(A0,requires_grad = False)
        mask = np.zeros((self.n_x,self.n_x))
        mask[:,0] = 1
        mask[:,-1] = 1
        mask[0,:] = 1
        mask[-1,:] = 1
        self.mask = mask>0.5
        #self.padding = int((kernel_size - 1) / 2)
        #self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
        #    1, padding=0, bias=False)

        # Fixed gradient operator
        #self.filter.weight = nn.Parameter(torch.FloatTensor(self.DerFilter), requires_grad=False)  

    def forward(self, input):
        '''
        input size should be (batch, x_size, y_size , 3)
        input[:,:,0] is the current solution u(x,y,t)
        input[:,:,1] is the current action f(x,y,t), which is  the old action
        input[:,:,2] is the new action f(x,y,t+dt) 
        '''
        batch = input.shape[0]
        x1 = input[:,:,:,0].reshape(batch,-1,1)
        f0 =  input[:,:,:,1].reshape(batch,-1,1)
        f1 =  input[:,:,:,2].reshape(batch,-1,1)
        b = torch.matmul(self.A0,x1) + (f0/2+f1/2)*self.dt
        out = torch.linalg.solve(self.A1, b).reshape(batch, self.n_x,self.n_x)
        
        '''
        Using mask to implement Dirichlet boundary condition 
        '''
        
        mask = torch.tensor(self.mask).to(self.device)

        output  = out.masked_fill(mask,0)

        

        
        return output




