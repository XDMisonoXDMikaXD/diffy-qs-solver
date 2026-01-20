import math
import numpy as np
import matplotlib.pyplot as plt

#A Primer in Numerics: Ordinary Differential Equations

#choose step size 
dt = 0.1 
#choose interval
T = 10

#set up parameters
w = 2*math.pi #natural frequency
z = 0.25      #damping coefficient
A = np.array([[0,1],
             [-w**2, -2*z*w]]) #spring-mass-damp system

x0 = np.array([2,1]) #initial conditions

print(x0)



