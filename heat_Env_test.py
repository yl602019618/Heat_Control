from heat_Environment import Heat_Env
import matplotlib.pyplot as plt
from fenics import plot
import numpy as np
env = Heat_Env()
obs = env.reset()
ob, reward, episode_over, _ = env.step(np.ones(6))

print(ob,reward,episode_over)

ob, reward, episode_over, _ = env.step(np.ones(6))

print(ob,reward,episode_over)

env.solver.update_plot(env.fig,env.solver.u_new)
 