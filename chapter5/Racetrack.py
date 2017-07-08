import sys,os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%matplotlib inline

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
'''
race_map = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
])
'''


start_line = [(12,5),(12,4),(12,3),(12,2)]
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



epsilon = 0.005

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

def pass_finishLine(pre_pos,lat_pos):
	finishLine_lx = finish_line[0][0]
	finishLine_hx = finish_line[-1][0]
	finishLine_y  = finish_line[0][1]

	if pre_pos[1] < finishLine_y and lat_pos[1] >= finishLine_y and (pre_pos[0] + lat_pos[0])/2 \
		<= finishLine_hx and (pre_pos[0] + lat_pos[0])/2 >= finishLine_lx:
		return True
	else:
		return False
		
def avg_return_per_episode(ep):
	ep_length = len(ep)
	ep_return = 0.0
	for i in range(2,ep_length,3):
		ep_return += ep[i]
	return ep_return*1.0 / ep_length		

def episode_generator():
	
	start_pos = start_line[random.randint(0,len(start_line)-1)]
	end_pos = start_pos
	last_pos = start_pos
	
	start_state = (start_pos[0],start_pos[1],start_velocity[0],start_velocity[1])	
	c_state = start_state
	
	episode = []
	episode.append(c_state)
	
	#print("start_state:",c_state)
	n = 0
	
	while not pass_finishLine(last_pos,end_pos) and n < 200:
		
		last_pos = end_pos

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
		if x_state[0] < 0 or x_state[0] > 13 or x_state[1] < 0 or x_state[1] > 19 or race_map[x_state[0],x_state[1]]==0: 
			#print "stucking..."
			#print "stuck action:",c_action
			#print "stuck state:",x_state	
			tmp_pos = start_line[random.randint(0,len(start_line)-1)]
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
		print("end position:",end_pos)		
	
	print("episode generated!")
	return episode

# p --> pos of reward in pair ; n --> episode length
def calReturnOfOnePair(p,n,episode):
	r = 0
	r = r + episode[p]
	
	for i in range(p+3,n,3):
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
	print("update done!")
		

class agent:
	
	cur_state = (9,5,0,0)
	traces = race_map
	
	
	def __init__(self,policy):
		self.policy = policy
	def forward(self):
		cur_action_opt = self.policy[self.cur_state]
		best_action_th = np.argmax([it[2] for it in cur_action_opt]) 
		best_action = (cur_action_opt[best_action_th][0],cur_action_opt[best_action_th][1])
		cur_v = (self.cur_state[2] + best_action[0],self.cur_state[3] + best_action[1])
		self.cur_state  = (self.cur_state[0]-cur_v[1],self.cur_state[1]+cur_v[0],cur_v[0],cur_v[1])
		self.traces[self.cur_state[0],self.cur_state[1]] = 2
	def stop(self):
		if (self.cur_state[0],self.cur_state[1]) in finish_line or race_map[self.cur_state[0],self.cur_state[1]] == 0:
			self.traces[self.cur_state[0],self.cur_state[1]] = 2
			return True
		else:
			return False



'''
fig = plt.figure()
ax = fig.add_subplot(111,autoscale_on=False,xlim=(0,race_map.shape[1]-1),ylim=(0,race_map.shape[0]-1))
ax.grid()
im = ax.imshow(np.flipud(race_map),origin='upper', interpolation='none')

anno_text = "Episode:%d,Timestep:%d,X_velocity:%d,Y_velocity:%d"
annotation = ax.annotate(anno_text %(0,0,0,0),xy=(5,11),bbox=
dict(boxstyle="round4,pad=0.3", fc="white", ec="b", lw=2))	
#annotation.set_animated(True)

#plt.show()
'''

def param_update():
	for eth,ep in enumerate(ep_list):
		for sth in range(0,len(ep),3):
			yield(ep[sth][0],ep[sth][1],sth/3,eth,ep[sth][2],ep[sth][3])
			#print ep[th]
		
def frame_update(step_info):
	x,y,sth,eth,x_v,y_v = step_info
	race_map_copy = np.copy(race_map)
	race_map_copy[x,y] = 4
	im.set_array(np.flipud(race_map_copy))
	annotation.set_text(anno_text % (eth,sth,x_v,y_v))
	return im,annotation


monte_carlo_num = 100
ep_list = []
avg_ep_return_list = []
f = open("tmp_data.txt",'w')
for i in range(monte_carlo_num):
	ep = episode_generator()
	print("episode length:%d" %(len(ep)/3))
	print("processing %d episode:" %i)
	cal_Q(ep)
	
	update_policy(ep)
	ep_list.append(ep)
	arpe = avg_return_per_episode(ep)
	avg_ep_return_list.append(arpe)
	f.write(('episode%d'%i) + 'return:' + str(arpe))
	f.write('\n')
f.close()

plt.title("assessment of racetrack problem")
plt.xlabel("episode index")
plt.ylabel("episode average return")
plt.plot(range(0,100),avg_ep_return_list,'r',label='avg return with on-policy')
plt.grid()

	
#anim = animation.FuncAnimation(fig, frame_update, frames=param_update, blit=False,save_count=9000)

plt.show()



'''
for i in range(monte_carlo_num):
	print("processing %d episode" %i)
	ep = episode_generator()
	cal_Q(ep)
	update_policy(ep)
	
ag1 = agent(policies)
stop_flag = ag1.stop()
while not stop_flag:
	ag1.forward()
print ag1.traces
#print policies
'''
