import torch
import torch.nn as nn
import gym
import Heat
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from main_PIPOMDP import heat_eqn

device = torch.device('cpu')
model = heat_eqn(exp_step = 300, total_step = 10000,update_freq_model = 10 , update_freq_policy = 10, device = device)
model.init_variable()
print(model.optimizer)
model.exp_step()
print('ptr',model.data_real.ptr)
print('ptr',model.data_real.ptr,'obs:',model.data_real.obs_buf[model.data_real.ptr-1],'act:',model.data_real.act_buf[model.data_real.ptr-1])
print('obs_shape',model.data_real.obs_buf.shape) # 500,10,10

a,xy ,a_c= model.train_SFNO_test(n_step = 5,batch_size=20)
print(model.grid.shape) # batch_size , n_step-1 , gridx, gridy , 2
a_c = a_c.numpy()
a = a.numpy()
xy = xy.numpy()
X = xy[:,:,0]
Y = xy[:,:,1]
a_true = a_c[0] + a_c[1]*X+a_c[2]*Y +a_c[3]*X*X+a_c[4]*Y*Y+ a_c[5]*X*Y

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,a)

fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,a_true)


