"""
New Main: ODE test 
@author: Cameron
"""

import numpy as np
from math import sqrt

import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

from OrbitPropagator import OrbitPropagator as OP
import Planetary_Data as pd
import tools as t

cb = pd.earth

#timespan
tspan = 5000 #100 min * 60 s/min -> s
dt = 100

if __name__ == '__main__':
    #a,e,i,ta,aop,raan=coes
    
    #ISS
    c0 = [cb['radius']+414.0, 0.0006189,51.6393,0.0,234.1955,105.6372]
    #find the orbital period of the first axis
    a0 = c0[0]
    tspan0 = t.find_period(a0)
    
    #GEO
    #c1 = [cb['radius']+35800.0,0.0,0.0,0.0,0.0,0.0]
    
    # random
    c2 = [cb['radius']+700.0,0.0006189,51.6393,0.0,234.1955,105.6372]
    a2 = c2[0]
    tspan2 = t.find_period(a2)
    
    #create orbit propagator
    op0 = OP(c0,tspan0,dt,coes=True)
    #op1 = OP(c1,tspan,dt,coes=True)
    op2 = OP(c2,tspan2,dt,coes=True)
    
    plt.figure()
    plt.plot(op0.ta[0:25], op0.rs[0:25,0], 'b',
             op0.ta[0:25], op0.rs[0:25,1], 'r',
             op0.ta[0:25], op0.rs[0:25,2], 'k',)
    
    t.plot_n_orbits([op0.rs, op2.rs],labels=['ISS','Random'],
                    show_plot=True)