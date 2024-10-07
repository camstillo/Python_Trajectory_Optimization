"""
Optimal Trajectory Solver
By Cameron Castillo
---------------------------------------------
This program solves a dynamic system which finds optimal trajectories
and thrust profiles for 

First, the program uses a Orbit propagator to numerically solve for
an intial and final orbit given either TLEs or by defining orbital 
parameters. See "/Orbit_Propagator" folder for details on this 
solver. This is based on a project by Alonso Gonzales (YouTube:
https://www.youtube.com/@alfonsogonzalez-astrodynam2207)

Next, the program uses the GEKKO optimization library to set 
initial and final orbits from the output of the orbit propagator. 
This results in a trajectory and thrust profile that was numerically
optimized for both time and fuel usage. The cost of fuel and time 
can be changed to optimize for one or the other, or get a solution 
in-between

Finally, the program plots the outputs of thrust and trajectory. See
tools.py in "/Orbit_Propagator" for a more in-depth description of the 
functions.
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.getcwd()
added_dir = current_dir + "//Orbit_Propagator"
sys.path.append(added_dir)

from Trajectory_Solver import traj_solver as TS
#from OrbitPropagator import OrbitPropagator as OP
import Planetary_Data as pd
import tools as t

'''
Parameters: 
Change this section to use different parameters in the simulation
---------------------------------------
min height      minimum orbital height, just keep at 300 unless already
                in a higher orbit [km]
max height      maximum orbital height, prevents simulation from going 
                thousands of km out [km]
n_steps         number of steps for solver. Will increase time to simulate
max_u           maximum thrust, depends on thruster. In the future, will
                add a function that takes in the specific impulse. [N]
time_cost       cost value of time. Increase to make time weighted more 
                heavily in the optimization (i.e. minimize time more than
                fuel or other parameter). Increasing too much will result
                in strange trajectories.
fuel_cost       cost value of fuel in optimization. Same as time_cost.
num_orbits      maximum number of orbital rotations. Just puts a final time
                parameter on the function even though it's minimizing for it
init_percent    initial position of orbit as percent of orbital period. 50% 
                will start 180 degrees from the starting point. This method 
                will be changed in the future. 
fin_percent     final index percentage, same as init_percent
cb              Central body, see planetary data folder for dictionary 
                entry. Includes 'radius,' 'mu,' and 'mass' for the body
dt              time step for Orbit propagator [s]. 
c0              initial orbital parameters in keplerian elements. Can use
                t.tle2coes to solve from TLE text file. order of elements 
                follows [a,e,i,ta,aop,raan]
c2              Final orbit, defined the same as c0
'''

#decay orbit
n_steps     = 101       # number of steps in soln
max_u       = 1e-6      # Max acceleration from thrust (m/s^2)
time_cost   = 0         # time cost
fuel_cost   = 1         # fuel cost
num_orbits  = 100       # max orbit times

init_percent = 50       # inital percent along orbit
fin_percent  = 5        # final percent along orbit


#propogation time span and step
cb = pd.earth   #sets central body as Earth
dt = 100        #time step for orbit propagator clas

#define initial and final orbit conditions
# [a,e,i,ta,aop,raan]
#ISS
#c0 = [cb['radius']+414.0, 0.01,51.6393,0.0,234.1955,105.6372]
a1,e1,i1,ta1,aop1,raan1 = t.tle2coes(current_dir + '//Dummy_TLEs//AO-85.txt')
c0 = [a1,e1,i1,ta1,aop1,raan1]

# random 
#c2 = [cb['radius']+900.0, 0.01,51.6393,0.0,234.1955,105.6372]
a,e,i,ta,aop,raan = t.tle2coes(current_dir + '//Dummy_TLEs//AO-85.txt')
c2 = [cb['radius'],0,i+90.0,ta,aop,raan]

min_height  = 300       # min orbit bound
max_height  = 2000      # max orbit bound 

toggle_j2 = True
toggle_aero = False
'''
END PARAMETERS
--------------------------------
If you're just playing with the program, you shouldn't need to change 
anything past here.
'''

solver = TS(c0, c2, init_percent, fin_percent, max_height, n_steps, 
            max_u, toggle_aero=toggle_aero,toggle_j2=toggle_j2)
rs, thrust, tm = solver.find_traj_gen()

#Set dark background
plt.style.use('dark_background')

#make thrust plots
#tm_adj = m.time * tf.VALUE[0]
#fig1 = plt.figure(figsize=(32,8))
fig1, (ax0, ax1, ax2) = plt.subplots(1,3)

#xthrust, ythrust, zthrust = thrust[:][0:3].tolist()
#print(thrust.shape)

ax0.plot(tm, thrust[:,0], 'r')
ax0.set(xlabel='Time [s]', ylabel='Thrust [N]')
ax0.set_title('X Thrust')
ax0.label_outer()

ax1.plot(tm, thrust[:,1], 'b')
ax1.set_title('Y Thrust')
ax1.set(xlabel='Time [s]', ylabel='Thrust [N]')
ax1.label_outer()

ax2.plot(tm, thrust[:,2], 'm')
ax2.set_title('Z Thrust')
ax2.set(xlabel='Time [s]', ylabel='Thrust [N]')
ax2.label_outer()

#define fig object
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111,projection='3d')

# plot trajectory
ax.plot(solver.op0.rs[:,0],solver.op0.rs[:,1],solver.op0.rs[:,2],'r--', label='initial orbit')

ax.plot(solver.op1.rs[:,0],solver.op1.rs[:,1],solver.op1.rs[:,2],'m--', label='final orbit')

ax.plot(rs[:,0],rs[:,1],rs[:,2],'b', label='Trajectory')
ax.plot(solver.op0.rs[solver.start_index,0],
        solver.op0.rs[solver.start_index,1], 
        solver.op0.rs[solver.start_index,2],
        marker='o', color='w', label='Initial Position')

# plot central body
#_u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
#_x = cb['radius']*np.cos(_u)*np.sin(_v)
#_y = cb['radius']*np.sin(_u)*np.sin(_v)
#_z = cb['radius']*np.cos(_v)
#ax.plot_surface(_x, _y, _z, cmap='Blues')

#plot the x, y, z vectors
l = cb['radius'] * 2
x,y,z=[[0,0,0],[0,0,0],[0,0,0]]
u,v,w=[[l,0,0],[0,l,0],[0,0,l]]
ax.quiver(x,y,z,u,v,w,color='k')

max_val=6000

ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

ax.set_xlabel(['X (km)'])
ax.set_ylabel(['Y (km)'])
ax.set_zlabel(['Z (km)'])

ax.set_aspect('equal')

ax.set_title('Trajectory Plot')
plt.legend()

plt.show()
