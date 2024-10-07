# -*- coding: utf-8 -*-
"""
Test aero-drag and J2 perturbations with orbit propagator class before moving to GEKKO
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import Planetary_Data as pd
import tools as t
from OrbitPropagator import OrbitPropagator as OP
from OrbitPropagator import null_perts

import ThrusterDict as TD

#[tspan] = seconds
tspan = 3600*30*24 # 1 month
dt = 100

#turn thrust function on and off
n_steps = int(np.ceil(tspan/dt))
#thrust_function = np.ones((int(np.ceil(tspan/dt)), 1))
thrust_function = np.zeros((n_steps, 1))

#edit thrust values
N = n_steps//10
for i in range(N):
    for j in range(10):
        thrust_function[i*N:i*N+j] = 1
print(np.shape(thrust_function))

cb = pd.earth

thruster = TD.Enpropulsion_NanoR3

if __name__ == '__main__':
    #define perts dict
    perts = null_perts()
    perts['J2'] = True
    perts['aero'] = True
    perts['Cd'] = 2.2
    perts['A'] = (100e-6)**2 #/4.0 # km^2
    perts['thrust'] = thruster.MaxThrust
    perts['isp'] = thruster.Isp
    perts['thrust_direction'] = 1
    
    #initial mass of spacecraft
    mass0 = 3.0
    
    rp = 480 + cb['radius']
    ra = 500 + cb['radius']
    
    raan = 340.0
    i = 65.2
    aop = 58.0
    ta = 332.0
    
    a = (rp + ra)/2.0
    e = (ra - rp)/(ra + rp)
    
    state0 = [a,e,i,ta,aop,raan]
    #state0 = [cb['radius']+800,0.1,10.0,0.0,0.0,0.0]
    
    op = OP(state0, tspan, dt, deg=True, coes=True, mass0=mass0, perts=perts,
            thruster=thruster, ignore_fuel_mass=False, thrust_function=thrust_function)
    #op.plot_mass(show_plot=True, hours=True)
    op.plot_alts(show_plot=True, hours=True)
    op.plot_3d(show_plot=True)
    op.calculate_coes()
    op.plot_coes(show_plot=True, hours=True)
    op.calc_apoapse_periapse()
    op.plot_apoapse_periapse(show_plot=True, hours=True)
    

