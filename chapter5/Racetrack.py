import sys,os
import numpy as np
import random


race_map = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
		     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
		     [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
	             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
                     [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
		     [0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0],
		     [0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0],
		     [0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
		     [0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

start_line = [(8,2),(8,3),(8,4),(8,5)]
finish_line = [(5,16),(6,16),(7,16)]

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
		tmp.append([ax,ay,1.0/actions_num])
	policies[(i,j,m,k)] = tmp

Q = dict()
for i,j,m,k in states:
	for ax,ay in actions:
		Q[((i,j,m,k),(ax,ay))] = 0

Returns = dict()
for i,j,m,k in states:
	for ax,ay in actions:
		Returns[((i,j,m,k),(ax,ay))] = list()



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
	episode.append(c_state)
	#print("start_state:",c_state)
	while end_pos not in finish_line:
		
		action_list = actions
		action_prob = [item[2] for item in policies[c_state]]
			
		c_action = random_pick(action_list,action_prob)
		
		# gurantee that velocity less than 5, more or equal 0
		c_velocity = (max(min(c_state[2]+c_action[0],4),0),max(min(c_state[3]+c_action[1],4),0))
		if c_velocity[0] == 0 and c_velocity[1] == 0:
			continue 
		
		# unsure state remaining to be justified
		x_state = (c_state[0]-c_velocity[1],c_state[1]+c_velocity[0],c_velocity[0],c_velocity[1])
		
		
		# if the car crash to wall, send it back at random start pos
		if x_state[0] < 0 or x_state[0] > 9 or x_state[1] < 0 or x_state[1] > 19: 
			#print "stucking..."
			#print "stuck action:",c_action
			continue
		elif race_map[x_state[0],x_state[1]] == 0:
			tmp_pos = start_line[random.randint(0,len(start_line)-1)]
			c_state = (tmp_pos[0],tmp_pos[1],0,0)
			c_reward = -5
		else:
			c_state = x_state
			c_reward = -1
		
		episode.append(c_action)
		episode.append(c_reward)
		episode.append(c_state)

		#print("action:",c_action)
		#print("next_state:",c_state)
		end_pos = (c_state[0],c_state[1])
		
	print("episode generated!")
	return episode
# p --> pos of reward in pair ; n --> episode length
def calReturnOfOnePair(p,n,episode):
	r = 0
	r = r + episode[p]
	
	for i in range(p,n,3):
		#print episode[i]
		r += episode[i]	
		
	return r

def cal_Q(episode):
	checked_pair = set()
	e_length = len(episode)
	if not e_length:
		print ("episode is empty!")
	else:
		# e_length-1 is for the omitting of terminal state, avoid the out of bound when episode[i+1]
		for i in range(0,e_length-1,3):
			sa_pair = (episode[i],episode[i+1])
			if sa_pair not in checked_pair:
				Returns[sa_pair].append(calReturnOfOnePair(i+2,e_length,episode)) 
				checked_pair.add(sa_pair)
				Q[sa_pair] = sum(Returns[sa_pair]) * 1.0 / len(Returns[sa_pair])					
	
	print("calculate Q done!")	

def update_policy(episode):
	tmpList_sa = []
	checked_state = set()
	e_length = len(episode)
	for i in range(0,e_length,3):
		s = episode[i]
		if s not in checked_state:
			for key in Q.keys():
				if key[0] == s:
					tmpList_sa.append((key[0],key[1],Q[key]))		
			best_action = tmpList_sa[np.argmax([it[2] for it in tmpList_sa])][1]
			# print best_action
			for a in policies[s]:
				if (a[0],a[1]) == best_action:
					a[2] = 1 - epsilon + epsilon / len(policies[s])
				else:
					a[2] = epsilon / len(policies[s])
		i = i + 3
	print("update done!")
		



if __name__ == '__main__':
	monte_carlo_num = 100
	for i in range(monte_carlo_num):
		print("processing %d episode" %i)
		ep = episode_generator()
		cal_Q(ep)
		update_policy(ep)
	
	print policies
