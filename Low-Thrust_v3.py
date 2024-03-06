"""
Copied from stack overflow post
Cameron Castillo
"""
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from OrbitPropagator import OrbitPropagator as OP
import Planetary_Data as pd
import tools as t

cb = pd.earth
#propogation time span and step
#tspan = 30*24*3600
dt = 100

#define initial and final orbit conditions
#ISS
c0 = [cb['radius']+414.0, 0.0006189,51.6393,0.0,234.1955,105.6372]
tspan0 = t.find_period(c0[0])

# random
c2 = [cb['radius']+700.0,0.0006189,51.6393,0.0,234.1955,105.6372]
tspan2 = t.find_period(c2[0])

#global timespan
tspan = 0
if tspan0 > tspan2:
    tspan = 2*tspan0
else:
    tspan = 2*tspan2

op0 = OP(c0,tspan0,dt,coes=True)
op1 = OP(c2,tspan2,dt,coes=True)

#Define a bunch of parameters
n_steps = 101         # number of steps in soln
max_u   = 100e-3      # Max thrust (N)
mu      = 3.986e14    # gravitational parameter earth (m**3/s**2)
tof     = tspan       # final time parameter

m = GEKKO(remote=False)

#define time varible
m.time = np.linspace(0, tof, n_steps)

#define fixed variables
theta_i = m.FV(lb = 0, ub = 360)
theta_f = m.FV(lb = 0, ub = 360)

rx_i = m.Var()
ry_i = m.Var()
rz_i = m.Var()

vx_i = m.Var()
vy_i = m.Var()
vz_i = m.Var()

rx_f = m.Var()
ry_f = m.Var()
rz_f = m.Var()

vx_f = m.Var()
vy_f = m.Var()
vz_f = m.Var()

#o1_tspan = m.Var()
#o2_tspan = m.Var()

ta_init = np.rad2deg(op0.ta)
ta_final = np.rad2deg(op1.ta)

#t_vect0 = np.linspace(0,tspan0,len(op0.rs))
#t_vect1 = np.linspace(0,tspan2,len(op1.rs))

m.cspline(theta_i, rx_i, ta_init, op0.rs[:, 0])
m.cspline(theta_i, ry_i, ta_init, op0.rs[:, 1])
m.cspline(theta_i, rz_i, ta_init, op0.rs[:, 2])

m.cspline(theta_i, vx_i, ta_init, op0.vs[:, 0])
m.cspline(theta_i, vy_i, ta_init, op0.vs[:, 1])
m.cspline(theta_i, vz_i, ta_init, op0.vs[:, 2])

m.cspline(theta_f, rx_f, ta_final, op1.rs[:, 0])
m.cspline(theta_f, ry_f, ta_final, op1.rs[:, 1])
m.cspline(theta_f, rz_f, ta_final, op1.rs[:, 2])

m.cspline(theta_f, vx_f, ta_final, op1.vs[:, 0])
m.cspline(theta_f, vy_f, ta_final, op1.vs[:, 1])
m.cspline(theta_f, vz_f, ta_final, op1.vs[:, 2])

r1 = m.Var(rx_i)
r2 = m.Var(ry_i)
r3 = m.Var(rz_i)

r1dot = m.Var(vx_i)
r2dot = m.Var(vy_i)
r3dot = m.Var(vz_i)

u1 = m.MV(lb = -max_u, ub = max_u)
u1.STATUS = 1
u2 = m.MV(lb = -max_u, ub = max_u)
u2.STATUS = 1
u3 = m.MV(lb = -max_u, ub = max_u)
u3.STATUS = 1

m.Equation(r1.dt() == r1dot)
m.Equation(r2.dt() == r2dot)
m.Equation(r3.dt() == r3dot)

r = m.Intermediate(m.sqrt(r1**2 + r2**2 + r3**2))
v = m.Intermediate(m.sqrt(r1dot**2 + r2dot**2 + r3dot**2))

m.Equation(-cb['mu']*r1/r**3 == r1dot.dt() + u1)
m.Equation(-cb['mu']*r2/r**3 == r2dot.dt() + u2)
m.Equation(-cb['mu']*r3/r**3 == r3dot.dt() + u3)

#m.fix_final(r1, rx_f)
#m.fix_final(r2, ry_f)
#m.fix_final(r3, rz_f)
# 
#m.fix_final(r1dot, vx_f)
#m.fix_final(r2dot, vy_f)
#m.fix_final(r3dot, vz_f)

#m.Minimize(m.integral(u1**2 + u2**2 + u3**2))

final = np.zeros(len(m.time))
final[-1] = 1
final = m.Param(value=final)

print(final*r1.VALUE) 

m.Obj((final*r1 - op1.rs[0, 0])**2)
m.Obj((final*r2 - op1.rs[0, 1])**2)
m.Obj((final*r3 - op1.rs[0, 2])**2)

m.Obj((final*r1dot - op1.vs[0, 0])**2)
m.Obj((final*r2dot - op1.vs[0, 1])**2)
m.Obj((final*r3dot - op1.vs[0, 2])**2)

m.options.IMODE = 6
m.options.solver = 3
#m.options.ATOL = 1e-3
m.options.MAX_ITER = 5000
m.solve(disp=True)    # solve

#Set dark background
plt.style.use('dark_background')

#define fig object
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111,projection='3d')

# plot trajectory
ax.plot(op0.rs[:,0],op0.rs[:,1],op0.rs[:,2],'r--', label='initial orbit')

ax.plot(op1.rs[:,0],op1.rs[:,1],op1.rs[:,2],'m--', label='final orbit')

ax.plot(r1.VALUE,r2.VALUE,r3.VALUE,'b', label='Trajectory')
ax.plot(r1.VALUE[0], r2.VALUE[0], r2.VALUE[0], marker='o',
        color='w', label='Initial Position')

# plot central body
_u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
_x = cb['radius']*np.cos(_u)*np.sin(_v)
_y = cb['radius']*np.sin(_u)*np.sin(_v)
_z = cb['radius']*np.cos(_v)
ax.plot_surface(_x, _y, _z, cmap='Blues')

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

plt.show
