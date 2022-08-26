import gym
import Heat
from Buffer import ReplayBuffer
import torch
import numpy as np
obs_dim  =10
act_dim = 6
init_step = 100
buffer_init = ReplayBuffer( obs_dim = obs_dim, act_dim = act_dim, size = 500)
env = gym.make('Heat_d-v0')

for step in range(init_step):
    action = env.action_space.sample()   # 从动作空间中随机选取一个动作
    observation = torch.Tensor(env.get_value())
    observation1, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    observation1 = torch.Tensor(observation1)
    reward = torch.Tensor(np.array(reward))
    action1 = torch.Tensor(action)
    done1= torch.Tensor(np.array(done,dtype = np.float32))
    buffer_init.store(obs = observation , act = action1, rew = reward, next_obs = observation1,done = done1,store_size = 1)
    print(step)
    if done :
        env.reset()
env.close()