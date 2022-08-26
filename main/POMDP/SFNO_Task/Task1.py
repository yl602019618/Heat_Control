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
from POMDP_Task1 import heat_eqn
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

model.plot_task1()
model.plot_baseline()