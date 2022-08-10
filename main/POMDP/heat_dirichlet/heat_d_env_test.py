from heat_d_env import MyGeometry, MyFunctionSpace, MySolver  
import matplotlib.pyplot as plt
import numpy as np
geometry = MyGeometry(min_x = 0.0, max_x =1.0 , min_y = 0.0,max_y = 1.0)
function_space = MyFunctionSpace(geometry, )
solver = MySolver(geometry, function_space, params={'T': 1,'dt': 0.01,'dimx':20,'dimy':20})
geometry.generate()

solver.function_space.generate()

solver.generate_variable()

solver.init_solve()

solver.generate_grid()
a = np.zeros(6)
solver.step_forward(a)
reward = solver.get_reward()
data = solver.get_value()
print(data.shape)
print('reward',reward)

a = 0.4*np.ones(6)
solver.step_forward(a)
reward = solver.get_reward()
data = solver.get_value()
print(data.shape)
print('reward',reward)
#solver.timestepping()
#solver.generate_grid()
#a = solver.get_value()
#print(a.shape)