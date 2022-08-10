import torch
import torch.nn as nn
from model import FNO2d, Heat_forward
import gym
import Heat
from Buffer import ReplayBuffer
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader



class heat_eqn():
    def _init_(self,exp_step = 1000, total_step = 10000,update_freq_model = 10 , update_freq_policy = 10, device = None):
        '''
        current_step
        current_episode
        exp_step_num : number of step in exploration period
        total_step : number of step in real environment in the whole training procedure
        update_freq_model : number of step in whole 2ed period takes for 1 step transition model training 
        update_freq_policy: number of step in whole 2ed period takes for 1 step policy model training
        obs_dim : the dimension of observation data
        act_dim : the dimension of action
        mode1 : first kind modes number of SFNO
        mode2 : second kind modes number of SFNO
        width : width of SFNO
        resolution : the dimension size of hidden state recovered by SFNO in x\y axis
        dt : match dt in fenics simulation setting
        '''
        self. current_step = 0
        self.current_episode = 0
        self.exp_step_num = exp_step
        self.total_step = total_step
        self.update_freq_model = update_freq_model
        self.update_freq_policy = update_freq_policy
        self.obs_dim = [10,10]
        self.act_dim = 6 
        self.mode1 = 5
        self.mode2 = 5
        self.width = 20
        self.resolution = (self.obs_dim[0]-1)*4+1
        self.SFNO = FNO2d(self.mode1 , self.mode2,self.width, self.resolution)
        self.dt = 0.01
        self.heat_forward = Heat_forward(n_x =self.resolution ,dt = self.dt)
        #self.model_A = model_A()

        self.env =  gym.make('Heat_d-v0')
        
        self.device = device
        self.data_real = ReplayBuffer( obs_dim = self.obs_dim[0], act_dim = self.act_dim, size = 500)
    def init_variable(self):
        self.env.reset()
        self.optimizer = optim.Adam(self.SFNO.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.998)

    def exp_step(self):
        observation_new = self.env.get_data()  #get observe of step 0
        for n in range(self.exp_step):
            action = self.env.action_space.sample()
            observation = torch.Tensor(self.env.get_value())
            observation1, reward, done, info = self.env.step(action)  # 用于提交动作，括号内是具体的动作
            observation1 = torch.Tensor(observation1)
            reward = torch.Tensor(np.array(reward))
            action1 = torch.Tensor(action)
            done1= torch.Tensor(np.array(done,dtype = np.float32))
            self.data_real.store(obs = observation , act = action1, rew = reward, next_obs = observation1,done = done1,store_size = 1)
            if done:
                self.env.reset()
                self.episode +=1
    
    def loss_gen(self,output, truth, beta ):
        '''
        input of SFNO is the (batch_size , n_step,obs_dim,obs_dim)
        output of SFNO is (batch_size , 2,(obs_dim-1)*r+1,(obs_dim-1)*r+1)
        truth of SFNO is  (batch_size,2,obs_dim,obs_dim) which represent the corresponding point of HR data
        beta is the fraction between data loss and physic-informed loss
        '''
        MSE_loss = nn.MSELoss()
        L1_loss=  nn.L1Loss()
        r = (output.shape[2]-1)/(self.obs_dim[0] -1)
        data_loss = L1_loss(output[:,:,0::r,0::r],truth)
        step_forward = self.heat_forward(output[:,0,:,:])
        phy_loss = MSE_loss(step_forward,output[:,1,:,:])
        total_loss =  data_loss+ phy_loss*beta
        return total_loss

    def get_grid(self, shape):
        batch_size, n, size_x, size_y = shape[0], shape[1],shape[2],shape[3]

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1,1,size_x,1,1).repeat([batch_size,n,1, size_y,1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1,1,1, size_y, 1).repeat([batch_size,n,size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def action_grid(self,action,grid):
        '''
        input is the action batch : (batch_size , n_step-1,act_dim)
        grid: (batch_size,n_step-1,nx,ny,2)
        output should be the action function wrt grid (batch_size,n_step-1 , grid_x, grid_y)
        input type: a1+a2*x[0]+a3*x[1]+a4*x[0]*x[0]+a5*x[1]*x[1]+a6*x[0]*x[1]
        '''
        
        


    def train_SFNO(self,epoch,n_step,batch_size):
        '''
        1. sample batch from D_real
        2. Training SFNO via physic-informed loss and data loss

        epoch is the number of training step
        n_step is the n_step in sampling batch
        '''

        self.grid = self.get_grid([batch_size,n_step-1,self.obs_dim[0],self.obs_dim[1]])
        for i in range(epoch):
            sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = n_step)
            
            action = self.action_grid(sample[action],self.grid)

            



               


        
        
        
        

    

