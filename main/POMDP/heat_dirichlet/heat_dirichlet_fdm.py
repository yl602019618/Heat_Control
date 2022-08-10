import numpy as np
from numpy import sin,cos,pi,exp


def ture_u(x,y,t):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.exp(-t)

def init_u(x,y):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def source(x,y,t):
    k =1/16
    return (-1+k*8*np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.exp(-t)

n_x = 10
n_y = 10
domain = [0,1,0,1]
T = 1
n_t  = 100
x = np.linspace(domain[0],domain[1],n_x)
y = np.linspace(domain[2],domain[3],n_y)
t= np.linspace(0,T,n_t)
dx = (domain[1] - domain[0])/n_x
dy = (domain[3] - domain[2])/n_y
dt = T/n_t
alpha = 1/16
r = alpha*dt/dx/dx

U = np.zeros((len(x),len(y),len(t)))
node = (len(x))*(len(y))   #自由度个数

d1 = np.ones((node,))*(-2*r+1)
d2 = np.ones((node,))*(2*r+1)

d3 = np.ones((node - 1,))*(r/2)
d4 = np.ones((node - len(x),))*(r/2)

A1 = np.diag(d2) - np.diag(d3,-1) - np.diag(d3,1) - np.diag(d4,- len(x)) - np.diag(d4, len(x))
A0 =  np.diag(d1) + np.diag(d3,-1) + np.diag(d3,1) + np.diag(d4,- len(x)) + np.diag(d4, len(x))


#定义初值
X,Y = np.meshgrid(x,y)
U[:,:,0] = init_u(X,Y)


time = 0

for i in range(n_t-1):
    time = time+dt
    U_NEW = U[:,:,i].reshape(-1)   #展开为一个一维的向量
    f_old = source(X,Y,time-dt).reshape(-1)
    f_new = source(X,Y,time).reshape(-1)
    b = A0@U_NEW + (f_old/2+f_new/2)*dt
    U_NEW = np.linalg.solve(A1,b)  
    U[:,:,i+1] = U_NEW.reshape(n_x,-1)   #再用reshape改变回来
    
    #强制改变边界条件
    U[:,-1,i+1] = 0
    U[:,0,i+1] = 0
    U[-1,:,i+1] = 0
    U[0,:,i+1] = 0
#print(U[:,:,-1])
TURE_U = ture_u(X,Y,time)   #t =1 时刻真解
#print(TURE_U)

import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D

ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,TURE_U)

fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,U[:,:,-1])
