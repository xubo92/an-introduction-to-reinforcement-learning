import os,sys
import numpy as np
import matplotlib.pyplot as plt


class Line_Chart:
	def __init__(self,xdata,ydata,xlabel,ylabel,title):
		self.xdata = xdata
		self.ydata = ydata
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.title  = title
	def Draw_LineChart(self,label):
		
		plt.title(self.title)
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		
		plt.plot(self.xdata,self.ydata,'y',label=label)
		plt.grid()
		
		plt.show()		
