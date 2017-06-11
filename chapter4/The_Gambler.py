import os,sys
import numpy as np
import math
import matplotlib.pyplot as plt

# current problem
# <1> why the value would be over 2 if i add goal_reward in the function:expect_return
# <2> why i can't reproduce the result of the book



goal_reward = 1
other_reward = 0

state_num = 99

states = []
for i in range(state_num):
	states.append(i+1)


values = np.zeros(state_num)
policies = np.zeros(state_num)

head_prob = 0.4
back_prob = 0.6

DISCOUNT = 1.0

def get_actions(state):
	return_actions = []
	action_limit = min(state,100-state)
	for j in range(action_limit+1):
		return_actions.append(j)
	return return_actions

def expect_return(single_state,single_action,values):
	
	returns = 0.0
	next_win_state = single_state + single_action
	next_lose_state = single_state - single_action
	
	returns = head_prob * (DISCOUNT * (1.0 if next_win_state==100 else values[next_win_state-1])) \
		+ back_prob * (other_reward + DISCOUNT * (0 if next_lose_state==0 else values[next_lose_state-1])) 

	#returns = head_prob * ((goal_reward if next_win_state==100 else other_reward) + DISCOUNT * (1.0 if next_win_state==100 else values[next_win_state-1])) \
	#	+ back_prob * (other_reward + DISCOUNT * (0 if next_lose_state==0 else values[next_lose_state-1]))

	return returns


if __name__ == "__main__":
	
	while(True):
		
		Delta = 0.0
		theta = 1e-9
		
		for s in states:
			tmp_value = values[s-1]
			actions = get_actions(s)
			value_candidates = []
			for a in actions:
				  value_candidates.append(expect_return(s,a,values))
			
			values[s-1] = np.max(value_candidates)
			
			print values[s-1]	
			#Delta = np.max(Delta,np.abs(tmp_value-values[s-1]))
			Delta += np.abs(tmp_value-values[s-1])
		if Delta < theta:
			break	
	
	for s in states:
		actions = get_actions(s)
		value_candidates = []
		for a in actions:
			value_candidates.append(expect_return(s,a,values))
		
		policies[s-1] = actions[np.argmax(value_candidates)]	
		
	
	plt.figure(1)
	plt.xlabel('Capital')
	plt.ylabel('Value estimate')
	plt.plot(values)

	plt.figure(2)
	plt.scatter(states,policies)
	plt.xlabel('Capital')
	plt.ylabel('Final policy (stake)')
	plt.show()	

	
