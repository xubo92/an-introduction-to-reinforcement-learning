import os,sys
import numpy as np
import matplotlib.pyplot as plt
from env import *
from result_analysis import *
from monte_carlo import *
from td import *
from td_lambda import *
from DQN import *


if __name__ == '__main__':
	
		
	Lamborghini = RaceCar()
	Lam_states  = Lamborghini.get_states()
	Lam_actions = Lamborghini.get_actions()

	'''
	MC = MonteCarlo(Lam_states,Lam_actions)
	learning_type = 'on-policy'
	MC.set_policy(learning_type)
		
	avg_ep_return_list = MC.on_policy_learning(Lamborghini,4000,0.2,200,50)
	'''
	'''
	TD = TemporalDifference(Lam_states,Lam_actions)
	TD.set_policy('sarsa')
	avg_ep_return_list = TD.sarsa_learning(Lamborghini,3000,0.1,0.5,1,200,50)
	xdata = range(0,3000,50)
	ydata = avg_ep_return_list
	'''
	'''
	TD = TemporalDifference(Lam_states, Lam_actions)
	TD.set_policy('q-learning')
	avg_ep_return_list = TD.Q_learning(Lamborghini, 10000, 0.1, 0.5, 1, 200, 100)
	xdata = range(0, 10000, 100)
	ydata = avg_ep_return_list
	'''
	'''
	td_lambda = Temporal_Difference_lambda(Lam_states,Lam_actions)
	td_lambda.set_policy('sarsa_lambda')
	avg_ep_return_list = td_lambda.sarsa_lambda(Lamborghini,2000,0.1,0.5,1,0.9,200,50)

	xdata = range(0,2000,50)
	ydata = avg_ep_return_list
	'''
	'''
	td_lambda = Temporal_Difference_lambda(Lam_states,Lam_actions)
	td_lambda.set_policy('naive_Q_lambda')
	avg_ep_return_list = td_lambda.sarsa_lambda(Lamborghini, 2000, 0.1, 0.5, 1, 0.9, 200, 50)

	xdata = range(0, 2000, 50)
	ydata = avg_ep_return_list

	fig = Line_Chart(xdata,ydata,"index","avg return","assessment of racetrack "+'with naive Q-lambda')
	fig.Draw_LineChart("")
	'''

	CartPole = CartPole()
	env = CartPole.get_env()

	DQN = DQN(env)
	DQN.Deep_Q_Learning(env,500,50000,32,0)

