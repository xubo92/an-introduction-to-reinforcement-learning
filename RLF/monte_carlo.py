import os,sys
import numpy as np
import matplotlib.pyplot as plt
from agent import *

class MoneCarlo:

	def __init__(self,state_list,action_list):
		self.states = state_list
		self.actions = action_list
		
		self.state_num = len(self.states)
		self.action_num = len(self.actions)

		self.Q = dict() 
		self.N = dict() 
		self.D = dict()
		
		for s in self.states:
			self.Q[s] = np.random.random(self.action_num)
			self.N[s] = np.zeros(self.action_num)
			self.D[s] = np.zeros(self.action_num)
	
	def set_policy(self,learning_type):
		self.pi = dict()
		self.mu = dict()
	
		if learning_type == 'off-policy':		
			for s in self.states:
				idx = np.random.randint(0,self.action_num,size=1)[0]
				self.pi[s] = np.zeros(self.action_num)
				self.pi[s][idx] = 1.0
				
				self.mu[s] = np.random.random(self.action_num)
				self.mu[s] = self.mu[s] / np.sum(self.mu[s])	
		elif learning_type == 'on-policy':
			pass
		else:
			pass

	def get_policy(self,policy_type):
		if policy_type == 'target policy':
			return self.pi
		elif policy_type == 'behavior policy':
			return self.mu			
		else:
			print("Get policy with error")			
			
	def off_policy_learning(self,agent,episode_num,epsilon,max_timestep,eval_interval):
			
		ep_idx = 0
		avg_ep_return_list = []
		while ep_idx < episode_num:
			
			
			if ep_idx % eval_interval == 0:
				eval_ep = agent.episode_generator(self.pi,max_timestep)
				print("eval episode length:%d" %len(eval_ep))
				c_avg_return = agent.avg_return_per_episode(eval_ep)
				avg_ep_return_list.append(c_avg_return)
				print("assessing return:%f" %c_avg_return)
			
			c_ep = agent.episode_generator(self.mu,max_timestep)
			ep_length = len(c_ep)
			
			print("processing the %dth episode:" %ep_idx)
			print("episode length:%d\n" %(ep_length/3))
			latest_time = ep_length - 3
			checked_sa = set()
			for i in range(ep_length-3,-1,-3):
				tmp_s = c_ep[i-1]
				if np.where(self.actions == c_ep[i]) == np.where(self.pi[tmp_s] == 1.0):
					continue
				else:
					latest_time = i
					print("latest_time:%d" %(i/3))
					break
			for j in range(latest_time+2,ep_length-3,3):
				if c_ep[j] not in checked_sa:
					checked_sa.add(c_ep[j])
					W = 1.0
					G = 0
					G += c_ep[j+2]
					sa_idx = np.where(self.actions == c_ep[j+1])
					for m in range(j+3,ep_length,3):
						W *= 1.0 / mu[c_ep[m]][np.where(self.actions == c_ep[m+1])]
	  					G += c_ep[m+2]
					self.N[c_ep[j]][sa_idx] += W * G	
					self.D[c_ep[j]][sa_idx] += W
					self.Q[c_ep[j]][sa_idx] = self.N[c_ep[j]][sa_idx] / self.D[c_ep[j]][sa_idx]	
			
			for s in self.states:
				best_action_idx = np.argmax(self.Q[s])
				#print("best-action:",self.actions[best_action_idx]) 
				self.pi[s] = [0.0] * self.action_num
				self.pi[s][best_action_idx] = 1.0
			
			ep_idx += 1
		return avg_ep_return_list

	def on_policy_learning():
		self.pi = dict()
		for s in self.states:
			self.pi[s] = np.random.random(self.action_num)



''' 
Lamborghini = RaceCar()
Lam_states  = Lamborghini.get_states()
Lam_actions = Lamborghini.get_actions()

MC = MoneCarlo(Lam_states,Lam_actions)
MC.set_policy('off-policy') 

ep = Lamborghini.episode_generator(MC.mu,200)
print ep
'''



