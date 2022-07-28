
from fenics import *
import mshr
import numpy as np
import matplotlib.pyplot as plt
from fenics import set_log_level, plot

class MyGeometry:
    def __init__(self,  min_x = 0.0, max_x =1.0 , min_y = 0.0,max_y = 1.0, params=None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        # self.W = W
        
        self.params = params
        # self.c

    
    def generate(self, params=None):

        channel = mshr.Rectangle(Point(self.min_x, self.min_y), Point(self.max_x, self.max_y))
        domain = channel 
        #self.mesh = mshr.generate_mesh(domain, 32)

        self.mesh = UnitSquareMesh(32, 32)
        bndry = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        for f in facets(self.mesh):
            mp = f.midpoint()
            if near(mp[0], self.min_x):  #left
                bndry[f] = 1
            elif near(mp[0], self.max_x):  # right
                bndry[f] = 2
            elif near(mp[1], self.max_y) :  # top
                bndry[f] = 3
            else :
                bndry[f] = 4
           

        
        self.bndry = bndry
        self.ds_left = Measure("ds", self.mesh, subdomain_data=self.bndry, subdomain_id=1)
        self.ds_right = Measure("ds", self.mesh, subdomain_data=self.bndry, subdomain_id=2)

        self.mesh_coor = self.mesh.coordinates()
        self.num_vertices = self.mesh.num_vertices()
        print('num of vertices',self.num_vertices)


class MyFunctionSpace:
    def __init__(self, geometry, params=None):
        self.geometry = geometry
        self.params = params
    
    def generate(self, params=None):
        self.V = FunctionSpace(self.geometry.mesh, "P", 1)
        

class MySolver:
    def __init__(self, geometry, function_space, params=None):
        self.geometry = geometry
        self.function_space = function_space
        self.params = params
        self.time = 0
        self.T = self.params['T']
        self.dt = self.params['dt']
        self.epoch = 0
        self.theta = 0.5
        self.u_0  =  Expression("x[0]", degree=1)

        
    def generate_variable(self):
        self.V = self.function_space.V
        self.ds_left = self.geometry.ds_left
        self.ds_right = self.geometry.ds_right
        

    def get_data(self,t,action,result = None):
        '''
        action is the vector that represent gaussian 

        '''
        f, g = result or (Constant(0), Constant(0))
        f.assign(2-t)
        g.assign(t)
        return f, g

    def create_timestep_solver(self, dsN, u_old):
    
        # Initialize coefficients
        self.f_n, self.g_n = self.get_data(0,action = ) # f  source term # g neumann boundary
        self.f_np1, self.g_np1 = self.get_data(0) 
        self.idt = Constant(0)
       
        # Extract function space
        V = self.function_space.V 

        # Prepare weak formulation
        self.u, self.v = TrialFunction(V), TestFunction(V)
        
        self.F = ( self.idt*(self.u - u_old)*self.v*dx
        + inner(grad(self.theta*self.u + (1-self.theta)*u_old), grad(self.v))*dx
        - (self.theta*self.f_np1 + (1-self.theta)*self.f_n)*self.v*dx
        - (self.theta*self.g_np1 + (1-self.theta)*self.g_n)*self.v*dsN
        )
        self.a, self.L = lhs(self.F), rhs(self.F)
    def solve(self,t,dt,action):
        self.get_data(t,result = (self.f_n,self.g_n),action = self.action_old)
        self.get_data(t+dt,result = (self.f_np1,self.g_np1),action = action)
        self.action_old = action 
        self.idt.assign(1/dt)
        old_level = get_log_level()
        warning = LogLevel.WARNING 
        set_log_level(warning)
        solve(self.a == self.L, self.u_new)
        set_log_level(old_level)

    def timestepping(self):
        self.u_new = Function(self.V)
        
        solver = self.create_timestep_solver( self.ds_left, self.u_new)
        self.u_new.interpolate(self.u_0)
        fig = self.init_plot()
        print("{:10s} | {:10s} | {:10s}".format("t", "dt", "energy"))
        t = 0
        while t < self.T:
            energy = assemble(self.u_new*dx)
            print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, self.dt, energy))
            self.solve(t, self.dt)
            t += self.dt
            self.update_plot(fig, self.u_new)

    def init_solve(self):
        self.u_new = Function(self.V)
        
        solver = self.create_timestep_solver( self.ds_left, self.u_new)
        self.u_new.interpolate(self.u_0)
        self.time = 0
    def step_forward(self,action):
        if self.time <self.T:
            energy = assemble(self.u_new*dx)
            print("{:10.4f} | {:10.4f} | {:#10.4g}".format(self.time, self.dt, energy))
            self.solve(self.time, self.dt,action)
            self.time += self.dt
       


    
    
    def init_plot(self):
        fig = plt.figure()
        fig.show()
        return fig

    def update_plot(self,fig, u, zlims=(0, 2)):
        fig.clear()
        p = plot(u)
        if p is None:
            return
        fig.colorbar(p)
        fig.canvas.draw()

    def generate_grid(self):
        gsx = self.params['dimx']
        gsy = self.params['dimy']
        xs = np.linspace(self.geometry.min_x, self.geometry.max_x, gsx)
        ys = np.linspace(self.geometry.min_y, self.geometry.max_y, gsy)
        mx, my = np.meshgrid(xs, ys)
        grids = np.stack((mx, my), 2)
        self.grids = grids
        self.meshgrid = [mx, my]

    def get_value(self):
        
        out = np.zeros(self.grids.shape[:2])
        for i in range(self.grids.shape[0]):
            for j in range(self.grids.shape[1]):
                xy = self.grids[i, j]
                out[i,j] = self.u_new(xy)
        return out 


    


    



        
       

   
