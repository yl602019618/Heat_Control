import numpy as np
from numpy import sin,cos,pi,exp
import torch 
from model import Heat_forward
def ture_u(x,y,t):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.exp(-t)

def init_u(x,y):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def source(x,y,t):
    k =1/16
    return (-1+k*8*np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.exp(-t)

n_x = 40
n_y = 40
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
fdm = Heat_forward(n_x =n_x,dt = dt , alpha = alpha)

batch = 3
U = torch.Tensor(np.zeros((batch,len(x),len(y),len(t),1)))

node = (len(x))*(len(y))   #自由度个数


#定义初值
X,Y = np.meshgrid(x,y)


U[:,:,:,0,0] = torch.Tensor(init_u(X,Y)).unsqueeze(0).repeat([batch,1,1])


time = 0

for i in range(n_t-1):
    time = time+dt
    f_old = torch.Tensor(source(X,Y,time-dt)).reshape([1,n_x,n_y,1]).repeat([batch,1,1,1])
    f_new = torch.Tensor(source(X,Y,time)).reshape([1,n_x,n_y,1]).repeat([batch,1,1,1])
    input = torch.cat((U[:,:,:,i,:],f_old,f_new),3)
    U[:,:,:,i+1,0] = fdm(input)
#print(U[:,:,-1])
TURE_U = ture_u(X,Y,1)   #t =1 时刻真解



U1 = U[2,:,:,-1,0].detach().numpy()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,U1)
plt.savefig('1.png')

fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.plot_surface(X,Y,TURE_U)
plt.savefig('2.png')