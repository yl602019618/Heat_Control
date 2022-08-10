from heat_dirichlet import MyGeometry, MyFunctionSpace, MySolver  
import matplotlib.pyplot as plt
import numpy as np
geometry = MyGeometry(min_x = 0.0, max_x =1.0 , min_y = 0.0,max_y = 1.0)
function_space = MyFunctionSpace(geometry, )
solver = MySolver(geometry, function_space, params={'T': 1,'dt': 0.01,'dimx':20,'dimy':20})
geometry.generate()

solver.function_space.generate()

solver.generate_variable()

solver.init_solve()
#solver.generate_grid()
solver.timestepping()
solver.generate_grid()
a,meshgird = solver.get_value()
b,meshgrid = solver.get_value1()
import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D

ax3d = Axes3D(fig)
ax3d.plot_surface(meshgird[0],meshgird[1],np.abs(a-b))
