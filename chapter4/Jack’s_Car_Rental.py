import sys,os
import numpy as np
import random
import copy
from math import *

# [1] Initialization
num_first = 21
num_second = 21

rent_reward = 10
move_reward = -2

DISCOUNT = 0.9


Poisson_request_first = 3
Poisson_return_first = 3
Poisson_request_second = 4
Poisson_return_second = 2


# Vital!!
Poisson_upbound = 11

Max_cars = 20
Max_moves = 5

num_states = num_first * num_second
value = np.zeros((num_first,num_second))
policy = np.zeros((num_first,num_second))

states = []
actions = []

for i in range(0,num_first):
	for j in range(0,num_second):
		states.append([i,j])

for i in range(-Max_moves,Max_moves+1,1):
	actions.append(i)

for i in range(0,num_first):
	for j in range(0,num_second):
		#policy[i][j] = random.randint(-5,5)
		policy[i][j] = 0



poisson_table = dict()
def poisson(n,lam):

	global poisson_table
	# to make no duplicate keys
	key = n * 10 + lam
	if key not in poisson_table.keys():
		poisson_table[key] = exp(-lam) * pow(lam,n) / factorial(n)

	return poisson_table[key]


def expect_return(single_state,single_action,value):
	
	returns = 0.0
	returns += move_reward * abs(single_action)


	for rental_request_first in range(0,Poisson_upbound):
		for rental_request_second in range(0,Poisson_upbound):

			NumOfCarsFirst = int(min(single_state[0]-single_action,Max_cars))
			NumOfCarsSecond = int(min(single_state[1]+single_action,Max_cars))

			realRentalFirst = min(NumOfCarsFirst,rental_request_first)
			realRentalSecond = min(NumOfCarsSecond,rental_request_second)

			NumOfCarsFirst -= realRentalFirst
			NumOfCarsSecond -= realRentalSecond

			reward = (realRentalFirst + realRentalSecond) * rent_reward
			prob = poisson(rental_request_first,Poisson_request_first) * \
					poisson(rental_request_second,Poisson_request_second)

			constant_return = True
			if constant_return:
				rental_return_first = Poisson_return_first
				rental_return_second = Poisson_return_second
				
				NumOfCarsFirst = min(NumOfCarsFirst + rental_return_first,Max_cars)
				NumOfCarsSecond = min(NumOfCarsSecond + rental_return_second,Max_cars)

				returns += prob * (reward + DISCOUNT * value[NumOfCarsFirst,NumOfCarsSecond])

			else:
				# vital!! temperary storage
				NumOfCarsFirst_ = NumOfCarsFirst
				NumOfCarsSecond_ = NumOfCarsSecond
				prob_ = prob

				for rental_return_first in range(0,Poisson_upbound):
					for rental_return_second in range(0,Poisson_upbound):

						NumOfCarsFirst = NumOfCarsFirst_
						NumOfCarsSecond = NumOfCarsSecond_
						prob = prob_

						NumOfCarsFirst = min(NumOfCarsFirst + rental_return_first,Max_cars)
						NumOfCarsSecond = min(NumOfCarsSecond + rental_return_second,Max_cars)

						prob = poisson(rental_return_first,Poisson_return_first) *\
								 poisson(rental_return_second,Poisson_return_second) * prob

						returns += prob * (reward + DISCOUNT * value[NumOfCarsFirst,NumOfCarsSecond])

	return returns





if __name__ == "__main__":

	
	theta = 1e-4
	newStateValue = np.zeros((Max_cars + 1, Max_cars + 1))
	while True:
					# [2] Policy Evaluation
		'''
		error = 0
		print "policy evaluation ..."
		for state in states:
			tmp = value[state[0],state[1]]
			value[state[0],state[1]] = expect_return(state,policy[state[0],state[1]],value)
			error = max(error,abs(tmp-value[state[0],state[1]]))
	
		print ("error:",error)
		if error > theta:
			continue
		'''

		for i, j in states:
			newStateValue[i,j] = expect_return([i, j], policy[i, j], value)
		error = np.sum(np.abs(newStateValue - value))
		print ("error:",error)
		if error >= 1e-4:
			value[:] = newStateValue
			continue
		value[:] = newStateValue

					# [3] policy improvement
		'''
		print "evaluation done"
		policy_stable = True

		print "policy improvement ..."
		for state in states:
			tmp = policy[state[0],state[1]]
			action_seq = []
			for a in actions:
				if (a > 0 and a < state[0]) or (a < 0 and abs(a) <= state[1]):
					action_seq.append(expect_return(state,a,value))
				else:
					action_seq.append(-float('inf'))

			best_action_th = np.argmax(action_seq)
			policy[state[0],state[1]] = actions[best_action_th]

			if tmp != policy[state[0],state[1]]:
				policy_stable = False
			
		
		if policy_stable:
			break
		else:
			print "need evaluation again ..."
		'''

		print "evaluation done"
		print "policy improvement ..."
		newPolicy = np.zeros((Max_cars + 1, Max_cars + 1))
		for state in states:
			action_seq = []
			for a in actions:
				if (a > 0 and a < state[0]) or (a < 0 and abs(a) <= state[1]):
					action_seq.append(expect_return(state,a,value))
				else:
					action_seq.append(-float('inf'))

			best_action_th = np.argmax(action_seq)
			newPolicy[state[0], state[1]] = actions[best_action_th]

		policyChanges = np.sum(newPolicy != policy)
		print('Policy for', policyChanges, 'states changed')
		if policyChanges == 0:
			policy = newPolicy
			break
		policy = newPolicy
		print ("need another evaluation...")


	print "optimal value function:",value
	print "optimal policy:",policy


















			
