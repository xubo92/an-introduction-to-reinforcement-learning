import os,sys
import numpy as np
import matplotlib.pyplot as plt
from agent import raceCar

class MoneCarlo:

	def __init__(self,state_list,action_list):
		self.states = state_list
		self.actions = action_list
		
		self.state_num = len(states)
		self.action_num = len(actions)

		self.Q = dict() 
		self.N = dict() 
		self.D = dict()
		
		for s in self.states:
			self.Q[s] = np.random.random(self.action_num)
			self.N[s] = np.zeros(self.action_num)
			self.D[s] = np.zeros(self.action_num)
			 
	def off_policy(self,episode_num,epsilon):
		self.pi = dict()
		self.mu = dict()
		
		for s in self.states:
			idx = np.random.randint(0,self.action_num,size=1)[0]
			self.pi[s] = np.zeros(self.action_num)
			self.pi[s][idx] = 1.0
			
			self.mu[s] = np.random.random(self.action_num)
		
		ep_idx = 0
		while ep_idx < episode_num:
			ep_idx += 1
			c_ep = raceCar.episode_generator()
			for i in range(len(c_ep)-3,-1,-3):
				tmp_s = c_ep[i-1]
				if c_ep[i] == pi[tmp_s][pi[tmp_s] == 1.0]:
					continue
				else:
					latest_time = i
					break
			
			
	def on_policy():
		self.pi = dict()
		for s in self.states:
			self.pi[s] = np.random.random(self.action_num)


