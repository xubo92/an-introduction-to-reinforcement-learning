import os,sys
import numpy as np
import matplotlib.pyplot as plt
from agent import *
from result_analysis import *
from monte_carlo import *



if __name__ == '__main__':
	
		
	Lamborghini = RaceCar()
	Lam_states  = Lamborghini.get_states()
	Lam_actions = Lamborghini.get_actions()

	MC = MonteCarlo(Lam_states,Lam_actions)
	MC.set_policy('on-policy') 
		
	avg_ep_return_list = MC.on_policy_learning(Lamborghini,500,0.2,200,50)
		
	xdata = range(0,500,50)
	ydata = avg_ep_return_list

	fig = Line_Chart(xdata,ydata,"index","avg return","assessment of racetrack off policy problem")
	fig.Draw_LineChart("")



