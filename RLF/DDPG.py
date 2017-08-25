import gym
import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten, Input, merge, Lambda


from keras.initializers import normal, identity
from keras import optimizers

class DDPG:

    def __init__(self,env):

        self.env = env

        self.TAU = 0.001
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.batch_size = 32
        self.gamma = 0.99


    def create_actor_network(self,state_shape,action_shape,h0_num,h1_num):

        if len(state_shape) < 2:
            model = Sequential()
            model.add(Dense(h0_num,activation='relu'))
            model.add(Dense(h1_num,activation='relu'))
            model.add(Dense(action_shape[0],activation='tanh'))
            model.compile(loss='mse',optimizer = optimizers.Adam(lr=self.actor_lr))
        else:
            print("Please check your state shape!")



    def create_critic_network(self):
        pass

