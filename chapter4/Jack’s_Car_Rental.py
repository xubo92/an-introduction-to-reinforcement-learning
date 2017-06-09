# coding: utf-8

# This is the code for programming exercises 4.5 called "Jack’s Car Rental" in book "reinforcement-learning-an-introduction"     #
# To better understand the code, please check Figure4.3 in chapter 4 for Pseudocode code					 #
# The code draw some reference to the code published at https://github.com/ShangtongZhang/reinforcement-learning-an-introduction #
# Author:Xubo Lv(lv_xubo@163.com) 2016.05                                                                                        #
##################################################################################################################################
import sys,os
import numpy as np
import random
import copy
from math import *

# [1] Initialization

# states num in location 1
num_first = 21
# states num in location 2
num_second = 21
# rent reward per car
rent_reward = 10
# move reward per car
move_reward = -2
#park reward for superflous cars
park_reward = -4

#discount parameter
DISCOUNT = 0.9

#poisson distribution expected value
Poisson_request_first = 3
Poisson_return_first = 3
Poisson_request_second = 4
Poisson_return_second = 2


# Vital!! when n > 11, the probility goes nearly to zero with the respect of the  expected value above
# so we don't care about what will happen when n > 11
Poisson_upbound = 11

# max cars limit
Max_cars = 20
# max moves limit
Max_moves = 5

# the number of states
num_states = num_first * num_second
# value matrix for every state
value = np.zeros((num_first,num_second))
# policy matrix for every state
policy = np.zeros((num_first,num_second))

states = []
actions = []

# initializiton of states
for i in range(0,num_first):
	for j in range(0,num_second):
		states.append([i,j])
# all possible actions
for i in range(-Max_moves,Max_moves+1,1):
	actions.append(i)
# arbitrary initial policy
for i in range(0,num_first):
	for j in range(0,num_second):
		#policy[i][j] = random.randint(-5,5)
		policy[i][j] = 0


# store every probability of every possible rent number or return number in avoid of duplicate calculation
poisson_table = dict()
def poisson(n,lam):

	global poisson_table
	# to make no duplicate keys
	key = n * 10 + lam
	if key not in poisson_table.keys():
		poisson_table[key] = exp(-lam) * pow(lam,n) / factorial(n)

	return poisson_table[key]


# core part: calculation of expected return for every possible state with possible actions
# the returns are used for update value matrix
def expect_return(single_state,single_action,value):
	
	returns = 0.0
	if (single_action >= 1):
		returns += move_reward * (single_action - 1)
	else:
		returns += move_reward * abs(single_action)
	
	_NumOfCarsFirst = int(min(single_state[0]-single_action,Max_cars))
	_NumOfCarsSecond = int(min(single_state[1]+single_action,Max_cars))
	
	if _NumOfCarsFirst > 10:
		returns += park_reward
	if _NumOfCarsSecond > 10:
		returns += park_reward 


	for rental_request_first in range(0,Poisson_upbound):
		for rental_request_second in range(0,Poisson_upbound):

			#NumOfCarsFirst = int(min(single_state[0]-single_action,Max_cars))
			#NumOfCarsSecond = int(min(single_state[1]+single_action,Max_cars))

			NumOfCarsFirst = _NumOfCarsFirst
			NumOfCarsSecond = _NumOfCarsSecond

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
				# vital!! temperary storage in avoid of the NumOfCarsFirst was modified by the following for loop
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
		
		# comment part is in-place version of policy iteration
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
			
		# comment part is in-place version of policy iteration
		'''
		print "evaluation done"
		policy_stable = True

		print "policy improvement ..."
		for state in states:
			tmp = policy[state[0],state[1]]
			action_seq = []
			for a in actions:
				if (a >= 0 and a <= state[0]) or (a < 0 and abs(a) <= state[1]):
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
				# here is where the former problem lies: a >= 0
				# if you don't be very careful about the conditions,then the ultimate result will contains a little bias
				if (a >= 0 and a <= state[0]) or (a < 0 and abs(a) <= state[1]):
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
