import torch
import torch.nn as nn
import gym
import Heat
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from
#  main_PIPOMDP import heat_eqn
from POMDP_now import heat_eqn
#from POMDP import heat_eqn
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device is ' , device)


model = heat_eqn(exp_step = 1500, total_step = 10000,update_freq_model = 10 , update_freq_policy = 10, device = device,n_step =10)
model.init_variable()
#print(model.optimizer.defaults['lr'])
model.exp_step()
#model.train_SFNO(epoch = 50000,n_step = 6,batch_size = 32)
model.train_SFNO(epoch = 60000,batch_size = 32)
model.plot_result()
#model.plot_result()
HR_error, Base_error, HR_error_l , Base_error_l,norm  = model.test_trajectory()

plt.figure()
t = np.linspace(0,1,90)
plt.plot(t,HR_error.cpu().detach().numpy(),label = 'HR_error')
plt.plot(t,Base_error.cpu().detach().numpy(),label = 'Base_error')
plt.plot(t,HR_error_l.cpu().detach().numpy(),label = 'HR_error_l')
plt.plot(t,Base_error_l.cpu().detach().numpy(),label = 'Base_error_l')
plt.plot(t,norm.cpu().detach().numpy(),label = 'norm')
plt.yscale('log')
plt.legend()
plt.savefig('error_of_traj.png')


error_data,error_phy,error_HR,norm_l = model.test_6pic()
plt.figure()
t = np.linspace(0,1,error_data.shape[0])
plt.plot(t,error_data.cpu().detach().numpy(),label = 'error_data')
plt.plot(t,error_phy.cpu().detach().numpy(),label = 'error_phy')
plt.plot(t,error_HR.cpu().detach().numpy(),label = 'error_HR')
plt.plot(t,norm_l.cpu().detach().numpy(),label = 'norm_l')
plt.yscale('log')
plt.legend()
plt.savefig('error_6.png')
