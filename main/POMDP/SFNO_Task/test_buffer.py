import torch
import torch.nn as nn
import gym
import Heat
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from main_PIPOMDP import heat_eqn
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device is ' , device)


model = heat_eqn(exp_step = 300, total_step = 10000,update_freq_model = 10 , update_freq_policy = 10, device = device)
model.init_variable()
model.exp_step()

sample = model.data_real.sample_batch_FNO(batch_size= 100,start = 0 , end = model.data_real.size,n_step = 5)
done = sample['done']
print(done)