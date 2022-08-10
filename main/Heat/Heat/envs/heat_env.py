import gym
from gym import spaces
from gym.utils import seeding
from gym import error, utils
import numpy as np
from os import path
import math

import torch
import random
import math
import torch.nn as nn

from Heat.envs.heat_d import *

import time

class Heat_d_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, params=None):
        self.set_params(params)


        self.action_space = spaces.Box(
            low= self.params['min_coef'],
            high=self.params['max_coef'],
            shape=(6,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low= self.params['min_obs'],
            high=self.params['max_obs'],
            shape=(self.params['dimx'], self.params['dimy']),  #(8*8)
            dtype=np.float32
        )
        self.geometry = MyGeometry(min_x = self.params['min_x'], max_x =self.params['max_x'] , min_y = self.params['min_y'],max_y = self.params['max_y'])
        self.function_space = MyFunctionSpace(self.geometry, )
        self.solver = MySolver(self.geometry, self.function_space, params={'T': self.params['T'] ,'dt': self.params['dt'],'dimx':self.params['dimx'],'dimy':self.params['dimy']})
        self.geometry.generate()

        self.solver.function_space.generate()
        self.solver.generate_variable()
        self.solver.init_solve()
        self.solver.generate_grid()


        self.current_t = self.solver.time 
        
    def set_params(self, params =None):
        if params is not None:
            self.params = params
        else:
            self.params = {'dt': 0.01,
                            'T': 1,
                            'dimx': 10,
                            'dimy': 10,
                            'min_x' : 0, 
                            'max_x' : 1, 
                            'min_y' : 0, 
                            'max_y' : 1 ,
                            'min_coef': -1,
                            'max_coef': 1,
                            'min_obs': -3,
                            'max_obs': 3
                                    }
        
    

    def step(self, action):
        time0 = time.time()
        self.solver.step_forward(action)
        #print(time.time()- time0)
        reward = self.solver.get_reward()
        #print(time.time()- time0)
        ob = self.solver.get_value()
        #print(time.time()- time0)
        episode_over = self._get_done()
        return ob, reward, episode_over, {}

    def reset(self):
        self.solver.init_solve()
        self.current_t = self.solver.time
        return self.solver.get_value()



    def _render(self, mode='grid', obj='u', close=False):
        if mode == 'grid':
            if obj  == 'u':
                pass
                
        elif mode == 'node':
            pass
 

    def _get_reward(self):
        
        return self.solver.get_reward()
        
 
    def _get_done(self):
        return self.current_t > self.solver.T
        # return self.env.getDown()
    def _close(self):
        return None

    def get_value(self):
        obs = self.solver.get_value()
        return obs 
