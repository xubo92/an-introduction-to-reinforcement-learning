import sys,os
import numpy as np
import random


race_map = np.array([[0,0,0,1,1,1,1,1,1,1],
		     [0,0,1,1,1,1,1,1,1,1],
	             [0,0,0,0,1,1,1,1,1,1],
	             [0,0,0,1,1,1,1,0,0,0],
                     [0,0,0,1,1,1,1,0,0,0],
		     [0,0,0,1,1,1,0,0,0,0],
		     [0,0,1,1,1,1,0,0,0,0],
		     [0,1,1,1,1,1,0,0,0,0],
		     [0,0,1,1,1,1,1,0,0,0],
		     [0,0,1,1,1,1,0,0,0,0]])

start_line = [(9,2),(9,3),(9,4),(9,5)]
finish_line = [(0,9),(1,9),(2,9)]

start_velocity = (0,0)

# states are the position of the car
states = []
rmx_n = race_map.shape[0]
rmy_n = race_map.shape[1]
vx_n  = np.arange(0,5).shape[0]
vy_n  = np.arange(0,5).shape[0] 
states_num = rmx_n * rmy_n * vx_n * vy_n
for i in range(rmx_n):
	for j in range(rmy_n):
		for m in range(0,5):
			for k in range(0,5):
				states.append((i,j,m,k))

actions = []
ax = np.arange(-1,2).shape[0]
ay = np.arange(-1,2).shape[0]
actions_num = ax * ay
for i in range(-1,2):
	for j in range(-1,2):
		actions.append((i,j))
	
policies = dict()
for i,j,m,k in states:
	tmp = []
	for ax,ay in actions:
		tmp.append((ax,ay,1.0/actions_num))
	policies[(i,j,m,k)] = tmp

Q = dict()
for i,j,m,k in states:
	for ax,ay in actions:
		Q[(i,j,m,k,ax,ay)] = 0

Returns = dict()
for i,j,m,k in start_line:
	for ax,ay in actions:
		Returns[(i,j,vx,vy,ax,ay)] = list()



epsilon = 0.05

def random_pick(some_list, probabilities):  
      x = random.uniform(0, 1)  
      cumulative_probability = 0.0  
      for item, item_probability in zip(some_list, probabilities):  
            cumulative_probability += item_probability  
            if x < cumulative_probability: break  
      return item

#some_list = [(1,1),(2,2)]
#prob = [0.6,0.4]
#print random_pick(some_list,prob)

'''
def find_nearest_pos(pos):
	pos_x = pos[0]
	pos_y = pos[1]
	
	for i in range(race_map.shape[1]):
		for j in range(i+1):
			if race_map[i,j]
'''
def episode_generator():
	
	start_pos = start_line[random.randint(0,len(start_line)-1)]
	end_pos = start_pos
	
	start_state = (start_pos[0],start_pos[1],start_velocity[0],start_velocity[1])	
	c_state = start_state
	
	episode = []
	episode.append(start_state)
	
	while end_pos not in finish_line:
		action_list = actions
		action_prob = [item[2] for item in policies[c_state]]
		
		c_action = random_pick(action_list,action_prob)
		
		# gurantee that velocity less than 5, more or equal 0
		c_velocity = (max(min(c_state[2]+c_action[0],4),0),max(min(c_state[3]+c_action[1],4),0)) 
		
		# unsure state remaining to be justified
		x_state = (c_state[0]+c_velocity[0],c_state[1]+c_velocity[1],c_velocity[0],c_velocity[1])
		
		# if the car crash to wall, send it back at random start pos 
		if race_map[x_state[0],x_state[1]] == 0:
			tmp_pos = start_line[random.randint(0,len(start_line)-1)]
			c_state = (tmp_pos[0],tmp_pos[1],0,0)
			c_reward = -5
		else:
			c_state = x_state
			c_reward = -1
		
		episode.append(c_action)
		episode.append(reward)
		episode.append(c_state)

	return episode
# p --> pos of reward in pair ; n --> episode length
def calReturnOfOnePair(p,n,episode):
	r = 0
	r = r + episode[p]
	
	for i in range(p,n):
		if i + 3 < n:
			i = i + 3
			r += episode[i]
		else:
			break		
		
	return r

def cal_Q(episode):

	e_length = len(episode)
	if not e_length:
		print ("episode is empty!")
	else:
		for i in range(e_length):
			sa_pair = (episode[i],episode[i+1])
			if sa_pair not in Returns:
				Returns[sa_pair].append(calReturnOfOnePair(i+2,e_length,episode)) 
				i = i + 3				
			

