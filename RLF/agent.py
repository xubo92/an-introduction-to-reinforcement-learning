import os,sys
import numpy as np

class RaceCar:

	global race_map 
	race_map = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
	            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,3,0,0,0],
	            [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0],
	            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0],
	            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
	            [0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0],
	            [0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0],
	            [0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	            [0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

	global start_line 
	start_line = [(12,5),(12,4),(12,3),(12,2)]
	global end_line 
	end_line   = [(5,16),(6,16),(7,16)]
	global start_velocity
	start_velocity = (0, 0)

	def __init__(self):
		self.race_map = race_map
		self.start_line = start_line
		self.end_line = end_line

		self.start_pos = (0,0)
		self.start_velocity = start_velocity

		self.c_state = (0,0,0,0)
		self.c_action = (0,0)
		self.c_reward = 0
		self.next_state = (0,0,0,0)

	def get_states(self):
		
		states = []
		rmx_n = self.race_map.shape[0]
		rmy_n = self.race_map.shape[1]
		vx_n  = np.arange(0,5).shape[0]
		vy_n  = np.arange(0,5).shape[0] 
		states_num = rmx_n * rmy_n * vx_n * vy_n
		for i in range(rmx_n):
			for j in range(rmy_n):
				for m in range(0,5):
					for k in range(0,5):
						states.append((i,j,m,k))
		return states
	
	def get_actions(self):
			
		actions = []
		ax = np.arange(-1,2).shape[0]
		ay = np.arange(-1,2).shape[0]
		for i in range(-1,2):
			for j in range(-1,2):
				actions.append((i,j))
		return actions
		
	def pass_endLine(self,pre_pos,lat_pos):
		
		endLine_lx = end_line[0][0]
		endLine_hx = end_line[-1][0]
		endLine_y = end_line[0][1]

		if pre_pos[1] < endLine_y and lat_pos[1] >= endLine_y and (pre_pos[0] + lat_pos[0])/2 \
                <= endLine_hx and (pre_pos[0] + lat_pos[0])/2 >= endLine_lx:
			return True
	
		else:
			return False
		
	
	def avg_return_per_episode(self,ep):
		
		ep_length = len(ep)
		ep_return = 0.0
		
		for i in range(2,ep_length,3):
			ep_return += ep[i]
		return ep_return * 1.0
		
	def episode_generator(self,policy,max_num,is_greedy):


		self.start_pos = self.start_line[np.random.randint(0,len(start_line)-1)]

		start_state = (self.start_pos[0],self.start_pos[1],self.start_velocity[0],self.start_velocity[1])
		self.c_state = start_state
		self.next_state = start_state
	
		episode = []
		episode.append(self.c_state)
	
		#print("start_state:",c_state)
		n = 0

		while not self.isTerminated(n,max_num):


			self.c_state = self.next_state

			action_list = self.get_actions()
			
			action_prob = policy[self.c_state]
			
			#print("action_list: ",action_list)
			#print("action_prob: ",action_prob)	

			if not is_greedy:
				self.c_action = action_list[np.random.choice(len(action_list),1,p=action_prob)[0]]
			else:
				self.c_action = action_list[np.argmax(action_prob)]

			self.c_state,self.c_action,self.c_reward,self.next_state = self.oneStep_generator()

			print "c_state:",self.c_state
			print "c_action:", self.c_action
			print "c_reward:", self.c_reward
			print "next_state:",self.next_state

			episode.append(self.c_action)
			episode.append(self.c_reward)
			episode.append(self.next_state)
			n += 1
			print "n:",n




		print("episode generated!")
		return episode


	def oneStep_generator(self):

		# return [s,a,r,s']



		self.next_state = (0, 0, 0, 0)
		self.c_reward = 0

		# gurantee that velocity less than 5, more or equal 0
		c_velocity = (max(min(self.c_state[2] + self.c_action[0], 4), 0), max(min(self.c_state[3] + self.c_action[1], 4), 0))

		x_state = (self.c_state[0] - c_velocity[1], self.c_state[1] + c_velocity[0], c_velocity[0], c_velocity[1])

		if x_state[0] < 0 or x_state[0] > 13 or x_state[1] < 0 or x_state[1] > 19 or self.race_map[
			x_state[0], x_state[1]] == 0:

			tmp_pos = self.start_line[np.random.randint(0, len(self.start_line) - 1)]
			self.next_state = (tmp_pos[0], tmp_pos[1], 0, 0)
			self.c_reward = -5

		elif c_velocity[0] == 0 and c_velocity[1] == 0:

			if np.random.choice(2, 1, p=[0.5, 0.5])[0] == 0:
				self.next_state = (x_state[0], x_state[1], 1, x_state[3])

			else:
				self.next_state = (x_state[0], x_state[1], x_state[2], 1)

			self.c_reward = -5

		else:
			self.next_state = x_state
			self.c_reward = -1

		return [self.c_state,self.c_action,self.c_reward,self.next_state]



	def getInitState(self):

		start_pos = self.start_line[np.random.randint(0, len(start_line) - 1)]
		return (start_pos[0],start_pos[1],self.start_velocity[0],self.start_velocity[1])

	def isTerminated(self,n,max_timestep):



		pre_pos = (self.c_state[0],self.c_state[1])
		lat_pos = (self.next_state[0],self.next_state[1])

		return self.pass_endLine(pre_pos,lat_pos) or n > max_timestep

