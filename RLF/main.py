import os,sys
import numpy as np
import matplotlib.pyplot as plt
from agent import *
from result_analysis import *
from monte_carlo import *
from td import *


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

	TD = TemporalDifference(Lam_states, Lam_actions)
	TD.set_policy('q-learning')
	avg_ep_return_list = TD.Q_learning(Lamborghini, 10000, 0.1, 0.5, 1, 200, 100)
	xdata = range(0, 10000, 100)
	ydata = avg_ep_return_list

	fig = Line_Chart(xdata,ydata,"index","avg return","assessment of racetrack "+'Q-learning')
	fig.Draw_LineChart("")



