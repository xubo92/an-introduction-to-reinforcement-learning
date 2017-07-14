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


	def __init__(self):
		self.race_map = race_map
		self.start_line = start_line
		self.end_line = end_line
	
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
		actions_num = ax * ay
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
		

	def episode_generator(self,policy,max_num):
	
			start_velocity = (0,0)			
			start_pos = self.start_line[np.random.randint(0,len(start_line)-1)]
			end_pos = start_pos
			last_pos = start_pos

			start_state = (start_pos[0],start_pos[1],start_velocity[0],start_velocity[1])	
			c_state = start_state
		
			episode = []
			episode.append(c_state)
		
			#print("start_state:",c_state)
			n = 0
		
			while not self.pass_endLine(last_pos,end_pos) and n < max_num:
				
				last_pos = end_pos

				action_list = self.get_actions()
				
				action_prob = policy[c_state]
				
				print("action_list: ",action_list)
				print("action_prob: ",action_prob)	
				
				c_action = action_list[np.random.choice(len(action_list),1,p=action_prob)[0]]
				
				# gurantee that velocity less than 5, more or equal 0
				c_velocity = (max(min(c_state[2]+c_action[0],4),0),max(min(c_state[3]+c_action[1],4),0))
				if c_velocity[0] == 0 and c_velocity[1] == 0:
					continue 
				
				# unsure state remaining to be justified
				x_state = (c_state[0]-c_velocity[1],c_state[1]+c_velocity[0],c_velocity[0],c_velocity[1])
				
				
				# if the car crash to wall, send it back at random start pos
				if x_state[0] < 0 or x_state[0] > 13 or x_state[1] < 0 or x_state[1] > 19 or self.race_map[x_state[0],x_state[1]]==0: 
					#print "stucking..."
					#print "stuck action:",c_action
					#print "stuck state:",x_state	
					tmp_pos = self.start_line[np.random.randint(0,len(self.start_line)-1)]
					c_state = (tmp_pos[0],tmp_pos[1],0,0)
					c_reward = -5
				else:
					c_state = x_state
					c_reward = -1
				
				episode.append(c_action)
				episode.append(c_reward)
				episode.append(c_state)
				n += 1
				#print("action:",c_action)
				#print("next_state:",c_state)
				end_pos = (c_state[0],c_state[1])
				#print("end position:",end_pos)		
			
			print("episode generated!")
			return episode

		
