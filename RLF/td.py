import os,sys
import numpy as np

class TD:
	def __init__(self,state_list,action_list,terminal_state):
		self.states = state_list
		self.actions = action_list

		self.terminal_state = terminal_state
	
		self.state_num = len(self.states)
		self.action_num = len(self.actions)

		for s in self.states:
			self.Q[s] = np.random.random(self.action_num)
		self.Q[self.terminal_state] = np.zeros((self.action_num))	
		

	def set_policy(self,learning_type):
		
		self.pi = dict()
		
		if learning_type == 'sarsa':
			for s in self.states:
				
		elif learning_type == 'q-learning':
			
