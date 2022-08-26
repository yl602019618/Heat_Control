import torch
import torch.nn as nn
#from model import FNO2d, Heat_forward
from model_with_batch import FNO2d, Heat_forward
import gym
import Heat
from Buffer import ReplayBuffer
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import wandb
wandb.init(project="POMDP")


class heat_eqn():
    def __init__(self,exp_step = 100, total_step = 10000,update_freq_model = 10 , update_freq_policy = 10,n_step = 3, device = None):
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
        super(heat_eqn, self).__init__()
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
        self.mode3 = 17
        self.mode4 = 17
        self.width = 20
        self.device = device
        self.resolution = (self.obs_dim[0]-1)*4+1
        self.scale =4
         # for n_step = 5
        self.n_step = n_step
        self.n = n_step+1
        self.SFNO = FNO2d(self.mode1 , self.mode2,self.mode3,self.mode4,self.width, self.resolution,self.n).to(self.device)
        self.dt = 0.01
        self.heat_forward = Heat_forward(n_x =self.resolution ,dt = self.dt, alpha =1/16,device = self.device).to(self.device)
        #self.model_A = model_A()
        self.beta_loss = [1,0.00002,0.3]
        self.env =  gym.make('Heat_d-v0')
        self.episode = 0
            
        wandb.config.n_step = n_step   
        self.data_real = ReplayBuffer( obs_dim = self.resolution, act_dim = self.act_dim, size = 1500,device = device)


    def init_variable(self):
        self.env.reset()
        self.optimizer = optim.Adam(self.SFNO.parameters(), lr=1e-3)
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.998)

    def exp_step(self):
        #observation_new = self.env.get_value()  #get observe of step 0
        for n in range(self.exp_step_num):
            action = self.env.action_space.sample()
            observation = torch.Tensor(self.env.get_value()).to(self.device)
            observation1, reward, done, info = self.env.step(action)  # 用于提交动作，括号内是具体的动作
            observation1 = torch.Tensor(observation1).to(self.device)
            reward = torch.Tensor(np.array(reward)).to(self.device)
            action1 = torch.Tensor(action).to(self.device)
            done1= torch.Tensor(np.array(done,dtype = np.float32)).to(self.device)
            self.data_real.store(obs = observation , act = action1, rew = reward, next_obs = observation1,done = done1,store_size = 1)
            if n % 100 == 0:
                print('current extrapolation step is :',n)
            if done:
                self.env.reset()
                self.episode +=1
    
    def loss_gen(self,output0,output1, truth, beta ,action):
        '''
        input of SFNO is the (batch_size , n_step,obs_dim,obs_dim)
        output0/1 of SFNO is (batch_size , (obs_dim-1)*r+1,(obs_dim-1)*r+1,1)
        truth of SFNO is  (batch_size,2,obs_dim,obs_dim) which represent the corresponding point of HR data
        beta_loss is the weight of data loss, boundary_loss and physic-informed loss
        '''
        beta1 = beta[0]
        beta2 = beta[1]
        beta3 = beta[2]
        MSE_loss = nn.MSELoss()
        #L1_loss=  nn.L1Loss()
        L1_loss= nn.MSELoss()
        #print((output0.shape[1]-1),(self.obs_dim[1]-1))
        r = (output0.shape[1]-1)//(self.obs_dim[0] -1)
        self.r = r
        #compute data_loss
        self.data_loss = L1_loss(output0[:,0::r,0::r,0],truth[:,0,:,:])+L1_loss(output1[:,0::r,0::r,0],truth[:,1,:,:])
        #boundary_loss 
        self.boundary_loss = (torch.norm(output0[:,[0,-1],:,0]) + torch.norm(output0[:,:,[0,-1],0]) +torch.norm(output1[:,[0,-1],:,0]) + torch.norm(output1[:,:,[0,-1],0]) )
        
        out_with_act = torch.cat((output0.permute(0,3,1,2),action),dim = 1).permute(0,2,3,1)
        #print('out_with_act shape',out_with_act.shape) #batch,3,n_x,n_x

        step_forward = self.heat_forward(out_with_act)
        #print('step_forward: ' , step_forward.shape)# batch, nx,ny
        #print('step_forward: ' , output1.shape)# batch, nx,ny,1
        #pde loss
        self.phy_loss = MSE_loss(step_forward,output1.permute(0,3,1,2)[:,0,:,:])
        self.total_loss = (self.data_loss*beta1 + self.boundary_loss*beta2 + self.phy_loss*beta3)
        return self.total_loss

    def get_grid(self, shape):
        batch_size, n, size_x, size_y = shape[0], shape[1],shape[2],shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(self.device)
        gridx = gridx.reshape(1,1,size_x,1,1).repeat([batch_size,n,1, size_y,1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float).to(self.device)
        gridy = gridy.reshape(1,1,1, size_y, 1).repeat([batch_size,n,size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def action_grid(self,action,grid):
        '''
        input is the action batch : (batch_size , n_step-1,act_dim)
        grid: (batch_size,n_step-1,nx,ny,2)
        output should be the action function wrt grid (batch_size,n_step-1 , grid_x, grid_y)
        input type: a1+a2*x[0]+a3*x[1]+a4*x[0]*x[0]+a5*x[1]*x[1]+a6*x[0]*x[1]
        '''
        action = action.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,grid.shape[2],grid.shape[3]])
        out = action[:,:,0,:,:]+ action[:,:,1,:,:]*grid[:,:,:,:,0]+action[:,:,2,:,:]*grid[:,:,:,:,1] \
            + action[:,:,3,:,:]*grid[:,:,:,:,0]*grid[:,:,:,:,0]+action[:,:,4,:,:]*grid[:,:,:,:,1]*grid[:,:,:,:,1]\
            +grid[:,:,:,:,1]*grid[:,:,:,:,0]*action[:,:,5,:,:]
        return out
        
    def downsample(self,input):
        return input[:,:,::4,::4]

    def train_SFNO(self,epoch,batch_size):
        '''
        1. sample batch from D_real
        2. Training SFNO via physic-informed loss and data loss

        epoch is the number of training step
        n_step is the n_step in sampling batch

        the action(t) followed the obs(t) represent act(x,y,t+dt),which is the source term of

        so heat forward should be act(t-1) obs(t) act(t)
        '''
        wandb.config.batchsize = batch_size
        wandb.config.epoch =epoch

        self.grid = self.get_grid([batch_size,self.n_step-1,self.obs_dim[0],self.obs_dim[1]]).to(self.device)
        self.grid_fine = self.get_grid([batch_size,2,self.resolution,self.resolution]).to(self.device)

        print_every = 10
        for i in range(epoch):
            sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = self.n_step)
            #action = self.action_grid(sample['act'],self.grid)
            # action is (batch_size,n_step-1 , grid_x, grid_y) 
            #obs is 
            obs = self.downsample(sample['obs']) #(batch_size,n_step ,grid_x,grid_y)
            input = obs[:,0:1,:,:]
            for j in range(self.n_step-1):
                #input = torch.cat((input,action[:,j:j+1,:,:]),dim = 1)
                input = torch.cat((input,obs[:,j+1:j+2,:,:]),dim = 1)
            # input (batch_size,2*n_step-1,grid_x,grid_y)
            #print('input_shape',input[:,:-2,:,:].shape)# 10,9,10,10 in test2
            output0 = self.SFNO(input[:,:-1,:,:].permute(0, 2, 3, 1))# batch , 1, x, y
            #print('output0.shape',output0.shape)
            output1 = self.SFNO(input[:,1:,:,:].permute(0, 2, 3, 1)) # batch , 1, x, y
            #need to add grid information to compute onestep heat forward
            # compute action of final t and t -1 in fine grid
            action_fine = self.action_grid(sample['act'][:,-2:,:],self.grid_fine)
            #print('action_fine shape',action_fine.shape)
            #print(input[0,[2*n_step-4,2*n_step-2],:,:])


            loss = self.loss_gen(output0 = output0 ,output1 = output1, truth = input[:,-2:,:,:],beta = self.beta_loss,action = action_fine)
            wandb.log({'loss_total': loss.item(), 'loss_data': self.data_loss.item(),'loss_boundary':self.boundary_loss.item(),'loss_physic':self.phy_loss.item(),'loss_physic_inside':self.phy_loss.item()*self.beta_loss[2],'loss_boundary_inside':self.boundary_loss.item()*self.beta_loss[1],'lr':self.optimizer.defaults['lr']})
            
            #loss.backward(retain_graph=True)
            loss.backward()
            
            #nn.utils.clip_grad_value_(self.SFNO.parameters(), clip_value=1.0)
            #self.optimizer.zero_grad()
            self.optimizer.step()
            #self.scheduler.step()
            self.optimizer.zero_grad()
            if i in [4000,7000,50000]:
                for params in self.optimizer.param_groups:
                    params['lr'] /= 2
                    wandb.log({'lr': params['lr']})
            if i in [10000,15000, 20000,30000,40000,45000]:
                for params in self.optimizer.param_groups:
                    params['lr'] /= 5
                    wandb.log({'lr': params['lr']})

            if (i+1) % print_every == 0:
                print('Epoch :%d ; Loss:%.8f;phy_loss %.8f;data_loss %.8f; boundary_loss %.8f;phy_loss_inside %.8f; boundary_loss_inside %.8f'  % (i+1, self.total_loss.item(),self.phy_loss.item(),self.data_loss.item(),self.boundary_loss.item(),self.phy_loss.item()*self.beta_loss[2],self.boundary_loss.item()*self.beta_loss[1]))
                #print('grad0:',self.SFNO.conv0.weights1.grad[0])
                #print('grad1:',self.SFNO.conv1.weights1.grad[0])


    def train_SFNO_test(self,batch_size):
        n_step= self.n_step
        self.grid = self.get_grid([batch_size,n_step-1,self.obs_dim[0],self.obs_dim[1]])
        sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = n_step)
        action = self.action_grid(sample['act'],self.grid)
        print(action.shape)
        action_new = sample['act']
        action_new = action_new[0:1,0:1,:]
        action_xy = self.action_grid(action_new,self.grid[0:1,0:1,:,:,:])
        return action_xy[0,0],self.grid[1,1] , action_new[0,0]



    def test_trajectory(self):
        '''
        1. sample one trajectory for prediction
        2. test from n_step to end
        3. compared with the error of direct prediction 
        '''
        self.data_test = ReplayBuffer( obs_dim = self.resolution, act_dim = self.act_dim, size = 100,device = self.device)
        self.env.reset()
        length = 100
        for n in range(length):
            action = self.env.action_space.sample()
            observation = torch.Tensor(self.env.get_value()).to(self.device)
            observation1, reward, done, info = self.env.step(action)  # 用于提交动作，括号内是具体的动作
            observation1 = torch.Tensor(observation1).to(self.device)
            reward = torch.Tensor(np.array(reward)).to(self.device)
            action1 = torch.Tensor(action).to(self.device)
            done1= torch.Tensor(np.array(done,dtype = np.float32)).to(self.device)
            self.data_test.store(obs = observation , act = action1, rew = reward, next_obs = observation1,done = done1,store_size = 1)
            if n % 10 == 0:
                print('current extrapolation step is :',n)
        '''
        SFNO input is size of Batch_size , n_step, x,y
        '''
        # length of data_store = 100 
        test_data = torch.zeros(length - self.n_step+1, self.n_step,self.resolution,self.resolution) 
        index = torch.arange(start =0,end = length - self.n_step+1)
        index = index.unsqueeze(-1)
        indexs = index.clone()
        for i in range(self.n_step-1):
            indexs = torch.cat((indexs,index+i+1),dim = 1)
        
        indexs = indexs.reshape(-1)
        input_HR = test_data.obs2_buf[indexs,:,:].reshape(-1,self.n_step,self.resolution,self.resolution)
        real_HR = test_data.obs_buf[indexs,:,:].reshape(-1,self.n_step,self.resolution,self.resolution) #batchsize , x,y
        output_HR = self.SFNO(self.downsample(input_HR[:,:,])) # batchsize ,x,y
        HR_error = torch.norm(output_HR - input_HR, p =1 ,dim = [1,2])/self.resolution**2
        Base_error = torch.norm(real_HR - input_HR , p = 1, dim = [1,2])/self.resolution**2

        HR_error_l = torch.norm(self.downsample(output_HR) - self.downsample(input_HR), p =1 ,dim = [1,2])/self.resolution**2
        Base_error_l = torch.norm(self.downsample(real_HR) - self.downsample(input_HR) , p = 1, dim = [1,2])/self.resolution**2

        return HR_error, Base_error, HR_error_l , Base_error_l


    def plot_result(self):
        n_step = self.n_step
        batch_size = 1
        self.grid = self.get_grid([batch_size,n_step-1,self.obs_dim[0],self.obs_dim[1]])
        self.grid_fine = self.get_grid([batch_size,2,self.resolution,self.resolution])
        sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = n_step)
        #action = self.action_grid(sample['act'],self.grid)
        obs = self.downsample(sample['obs']) #(batch_size,n_step ,grid_x,grid_y)
        input = obs[:,0:1,:,:]
        for j in range(n_step-1):
                #input = torch.cat((input,action[:,j:j+1,:,:]),dim = 1)
                input = torch.cat((input,obs[:,j+1:j+2,:,:]),dim = 1)
        output0 = self.SFNO(input[:,:-1,:,:].permute(0, 2, 3, 1))# batch , 1, x, y
        #print('output0.shape',output0.shape)
        output1 = self.SFNO(input[:,1:,:,:].permute(0, 2, 3, 1)) # batch , 1, x, y
        action_fine = self.action_grid(sample['act'][:,-2:,:],self.grid_fine)
        out_with_act = torch.cat((output0.permute(0,3,1,2),action_fine),dim = 1).permute(0,2,3,1)
        #print('out_with_act shape',out_with_act.shape) #batch,3,n_x,n_x
        step_forward = self.heat_forward(out_with_act)
        r = (output0.shape[1]-1)//(self.obs_dim[0] -1)
        '''
        input[0,2*n_step-4,:,:]
        output0[0,0,:,:] 
        output0[0,0,0::r,0::r]
        input[0,2*n_step-2,:,:]
        output1[0,0,:,:] 
        output1[0,0,0::r,0::r]
        '''
        # print LR u(t)
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        gridx = self.grid[0,0,:,:,0].cpu().detach().numpy()
        gridy = self.grid[0,0,:,:,1].cpu().detach().numpy()
        gridx_f = self.grid_fine[0,0,:,:,0].cpu().detach().numpy()
        gridy_f = self.grid_fine[0,0,:,:,1].cpu().detach().numpy()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,input[0,-2,:,:].cpu().detach().numpy())
        plt.title("LR_true_ut")
        plt.savefig('./img/LR_true_ut.png')
        wandb.log({"LR_true_ut": wandb.Image('./img/LR_true_ut.png')})

        #print HR predicted ut
        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx_f,gridy_f,output0[0,:,:,0].cpu().detach().numpy())
        plt.title("HR_predicted_ut")
        plt.savefig('./img/HR_predicted_ut.png')
        wandb.log({"HR_predicted_ut": wandb.Image('./img/HR_predicted_ut.png')})

        #print LR ut+1
        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,input[0,-1,:,:].cpu().detach().numpy())
        plt.title("LR_true_utp1")
        plt.savefig('./img/LR_true_utp1.png')
        wandb.log({"LR_true_utp1": wandb.Image('./img/LR_true_utp1.png')})
        #print HR predicted ut+1
        fig = plt.figure()
        ax3d = Axes3D(fig)
        #print('output1',output1[0,:,:,0])
      
        ax3d.plot_surface(gridx_f,gridy_f,output1[0,:,:,0].cpu().detach().numpy())
        plt.title("HR_predicted_utp1")
        plt.savefig('./img/HR_predicted_utp1.png')
        wandb.log({"HR_predicted_utp1": wandb.Image('./img/HR_predicted_utp1.png')})


        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx_f,gridy_f,step_forward[0,:,:].cpu().detach().numpy())
        plt.title("HR_predicted_utp1_p")
        plt.savefig('./img/HR_predicted_utp1_p.png')
        wandb.log({"HR_predicted_utp1_p": wandb.Image('./img/HR_predicted_utp1_p.png')})

        # print HR ut in LR grid
        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,output0[0,0::r,0::r,0].cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("HR_predicted_ut_LR")
        plt.savefig('./img/HR_predicted_ut_LR.png')
        wandb.log({"HR_predicted_ut_LR": wandb.Image('./img/HR_predicted_ut_LR.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,output1[0,0::r,0::r,0].cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("HR_predicted_utp1_LR")
        plt.savefig('./img/HR_predicted_utp1_LR.png')
        wandb.log({"HR_predicted_utp1_LR": wandb.Image('./img/HR_predicted_utp1_LR.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,torch.abs(output0[0,0::r,0::r,0]-input[0,-2,:,:]).cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("error_t")
        plt.savefig('./img/error_t.png')
        wandb.log({"error_t": wandb.Image('./img/error_t.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,torch.abs(output1[0,0::r,0::r,0]-input[0,-1,:,:]).cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("error_tp1")
        plt.savefig('./img/error_tp1.png')
        wandb.log({"error_tp1": wandb.Image('./img/error_tp1.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx,gridy,torch.abs(output1[0,0::r,0::r,0]-step_forward[0,0::r,0::r]).cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("error_tp1_pred")
        plt.savefig('./img/error_tp1p.png')
        wandb.log({"error_tp1_pred": wandb.Image('./img/error_tp1p.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx_f,gridy_f,torch.abs(output0[0,:,:,0]-sample['obs'][0,-2,:,:]).cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("error_bench_t")
        plt.savefig('./img/error_bench_t.png')
        wandb.log({"error_bench_t": wandb.Image('./img/error_bench_t.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx_f,gridy_f,torch.abs(output1[0,:,:,0]-sample['obs'][0,-1,:,:]).cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("error_bench_tp1")
        plt.savefig('./img/error_bench_tp1.png')
        wandb.log({"error_bench_tp1": wandb.Image('./img/error_bench_tp1.png')})

        fig = plt.figure()
        ax3d = Axes3D(fig)
        ax3d.plot_surface(gridx_f,gridy_f,sample['obs'][0,-2,:,:].cpu().detach().numpy())
        #print('step_forward',step_forward[0])
        plt.title("_bench_t")
        plt.savefig('./img/bench_t.png')
        wandb.log({"bench_t": wandb.Image('./img/bench_t.png')})