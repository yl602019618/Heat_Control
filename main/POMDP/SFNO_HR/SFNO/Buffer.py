import torch
import numpy as np
from torch.utils.data import Dataset
import gym
from utils import combined_shape, nptotorch

class ReplayBuffer(object):
    """
    First In First Out experience replay buffer agents.
    """

    def __init__(self, obs_dim, act_dim, size, device=None):
        '''
        obs_dim: observation data dimension
        act_dim :action_ dimension
        size: size of a buffer which control the sample number that restored in the buffer
        ptr : curren step ,which runs recurrently in the buffer
        '''

        super(ReplayBuffer,self).__init__()
        self.device = device
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.obs_buf = torch.zeros((size, obs_dim,obs_dim), dtype=torch.float) #zeros(size,obs_dim)
        self.obs2_buf = torch.zeros((size, obs_dim,obs_dim), dtype=torch.float) #zeros(size,obs_dim)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float)
        self.rew_buf = torch.zeros(size, dtype=torch.float)   #zeros(size)
        self.done_buf = torch.zeros(size, dtype=torch.float)  #zeros(done)
        if device is not None:
            self.obs_buf = self.obs_buf.to(device)
            self.obs2_buf = self.obs2_buf.to(device)
            self.act_buf = self.act_buf.to(device)
            self.rew_buf = self.rew_buf.to(device)
            self.done_buf = self.done_buf.to(device)
                
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done,store_size=1):
        self.obs_buf[self.ptr:self.ptr+store_size] = obs
        self.obs2_buf[self.ptr:self.ptr+store_size] = next_obs
        self.act_buf[self.ptr:self.ptr+store_size] = act
        self.rew_buf[self.ptr:self.ptr+store_size] = rew
        self.done_buf[self.ptr:self.ptr+store_size] = done
        self.ptr = (self.ptr+store_size) % self.max_size  # 如果超过maxsize就重写
        self.size = min(self.size+store_size, self.max_size) #现在的数据量 

    def sample_batch(self, batch_size=32,start=0,end=int(1e8)):
        '''
        sample from start to end
        '''
        idxs = torch.randint(start, min(self.size,end), size=(batch_size,))
        return dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

    def sample_batch_FNO(self,batch_size,start = 0 , end = int(1e8),n_step = 2):
        '''
        sample from start to end, each sample contains n_step obs-action and final obs2
        obs : obs0 ,... ,obs_(n_step-1)
        act : act0,act1,... , act(n_step-2)
        done: done0 , done1 , ... , done_(n_step-1)                 
        '''
        idxs = torch.randint(start, min(self.size-n_step+1,end), size=(2*batch_size,)).to(self.device) # batch_size*2
        
        idxs = idxs.unsqueeze(-1)# batch_size*2,1
        idx = idxs.clone()# batch_size*2,1


        for i in range(n_step-1):
            idxs = torch.cat((idxs,idx+i+1),dim = 1)
        # idxs batchsize ,n_step
        
        
        idx_obs  = idxs.reshape(-1)# 2*batchsize *n_step
        done_before_select=self.done_buf[idx_obs].reshape(2*batch_size,n_step)
        done_before_select = torch.sum(done_before_select,dim = 1)
        idxs = idxs[done_before_select<1,:]
        idxs = idxs[:batch_size,:]
        
        idx_obs = idxs.reshape(-1)

        idx_act = idxs[:,:-1].reshape(-1)
        obs = self.obs_buf[idx_obs].reshape(-1,n_step,self.obs_dim,self.obs_dim)
        act = self.act_buf[idx_act].reshape(-1 , n_step-1,self.act_dim)
        rew=self.rew_buf[idx_act].reshape(-1, n_step-1,1)
        done=self.done_buf[idx_obs].reshape(-1,n_step, 1)

        


        return  dict(obs=obs, act=act,rew=rew,done= done)


class Buffer_for_real(object):
    """
    First In First Out experience replay buffer agents.
    """

    def __init__(self, obs_dim, act_dim, size, device=None):
        super(ReplayBuffer,self).__init__()
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float)  #zeros(size,obs_dim)
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float) #zeros(size,obs_dim)
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float)
        self.rew_buf = torch.zeros(size, dtype=torch.float)   #zeros(size)
        self.done_buf = torch.zeros(size, dtype=torch.float)  #zeros(done)
        if device is not None:
            self.obs_buf = self.obs_buf.to(device)
            self.obs2_buf = self.obs2_buf.to(device)
            self.act_buf = self.act_buf.to(device)
            self.rew_buf = self.rew_buf.to(device)
            self.done_buf = self.done_buf.to(device)
                
        self.ptr, self.size= 0, size  # self.ptr indicates how many real data is stored

    def store(self, obs, act, rew, next_obs, done,store_size=1):
        self.obs_buf[self.ptr:self.ptr+store_size] = obs
        self.obs2_buf[self.ptr:self.ptr+store_size] = next_obs
        self.act_buf[self.ptr:self.ptr+store_size] = act
        self.rew_buf[self.ptr:self.ptr+store_size] = rew
        self.done_buf[self.ptr:self.ptr+store_size] = done
        self.ptr = self.ptr+store_size
        
    
    def sample_batch(self, batch_size=32,start=0,end=(1e8)):
        '''
        batch_size denotes the number of batch sampled this time
        start indicates the start index of data
        end indicates the end index of data which we usually take self.ptr 
        '''
        idxs = torch.randint(start, self.ptr, size=(batch_size,))
        return dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])


