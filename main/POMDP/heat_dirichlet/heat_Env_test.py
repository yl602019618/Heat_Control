from heat_d_Environment import Heat_d_Env
import matplotlib.pyplot as plt
from fenics import plot
import numpy as np
env = Heat_d_Env()
obs = env.reset()
ob, reward, episode_over, _ = env.step(np.ones(6))

print(reward,episode_over)
ob, reward, episode_over, _ = env.step(np.ones(6))

env.reset()
ob, reward, episode_over, _ = env.step(np.ones(6))

print(reward,episode_over)


 