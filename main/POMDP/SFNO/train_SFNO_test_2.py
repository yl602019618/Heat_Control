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
model.train_SFNO(epoch = 200,n_step = 5,batch_size = 20)
model.plot_result(n_step = 5)