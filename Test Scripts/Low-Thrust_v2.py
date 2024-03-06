# -*- coding: utf-8 -*-
"""
Low-Thrust_v2.py

Same as Low-Thrust_Optimizat.py but uses x and y coordinates 
as manipulated thrust variables instead of alpha and beta
angles. Can be transformed back to alpha and beta if soln
is more optimal.

Cameron Castillo
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

#Initialize Solver 
m = GEKKO(remote=False)

# Define Problem Constants
year_to_sec_conv = 365*24*3600   # s / yr
day_to_sec_conv  = 24*3600       # s / day
h_0       = 500e3                # initial orbital height (m)
h_f       = 550e3                # final orbital height (m)
r_earth   = 6.371e6              # earth radius (m)
r_0       = h_0 + r_earth        # initial radius (m)
v_tan_0   = 7.5969e3             # tangential velocty intial (m/s)
a_so      = 5.93e-3              # gravitational acc at Earth distance from sun (m/s)
h_min     = 200                  # min height (m)
light_num = 1.15e-1              # lightness number of sail (unitless)
mu_earth  = 3.986e14             # gravitational parameter earth (m**3/s**2)

T_max     = m.Const(100e-3)      # Maximum Thrust (N)
m_0       = m.Const(4)           # Wet Mass (kg)
m_prop    = m.Const(1)           # Propellant mass (kg)
I_sp      = m.Const(1000)        # Specific Impulse (s)
mu        = m.Const(mu_earth)    # Grav Parameter (kg**3/s**2)
g_0       = m.Const(9.81)        # Gravitational Accel (m/s**2)

#define variables
x   = m.Var(value=r_0)
y   = m.Var(value=0)
z   = m.Var(value=0)
dx  = m.Var(value=0)
dy  = m.Var(value=0)
dz  = m.Var(value=v_tan_0)
ddx = m.Var()
ddy = m.Var()
ddz = m.Var()
rad = m.Var(value=r_0, lb=h_min+r_earth)
mass = m.Var(value=m_0, lb=m_0 - m_prop, ub=m_0)

#define time variable
num_points = 101
tm = np.linspace(0,1,num_points)
m.time = tm # sec

#define Control Variables (MV)
T_x = m.MV(value=0, lb=0, ub=T_max)
T_x.STATUS = 1
T_y = m.MV(value=0, lb=0, ub=T_max)
T_y.STATUS = 1
T_z = m.MV(value=0, lb=0, ub=T_max)
T_z.STATUS = 0

#define fixed variables
tf = m.FV(value=1.0, lb=0.001, ub=10*year_to_sec_conv)
tf.STATUS = 1

#define variable relationships
m.Equations((dx==x.dt()/tf, dy==y.dt()/tf, dz==z.dt()/tf))
m.Equations((ddx==dx.dt()/tf, ddy==dy.dt()/tf, ddz==dz.dt()/tf))
m.Equation(rad == m.sqrt(x**2 + y**2 + z**2))

#define System dynamics
m.Equation(ddx == (-1*mu/rad**3)*x + (T_x/mass))
m.Equation(ddy == (-1*mu/rad**3)*y + (T_y/mass))
m.Equation(ddz == (-1*mu/rad**3)*z + (T_z/mass))
m.Equation(mass == m_0 - 
           (1/(g_0*I_sp))*m.integral(m.sqrt(T_x**2 + T_y**2 + T_z**2)))

#define final value
#m.fix_final(rad, val=h_f+r_earth)
final = np.zeros_like(m.time)
final[-1] = 1.0
final = m.Param(value=final)
#m.Equation(final*rad < r_earth+h_f)

#minimize final value
m.obj(final*(rad - (r_earth+h_f))**2)

#define objective fn
m.Obj(tf)

# Set solver mode
m.options.IMODE = 6

# Increase maximum number of allowed iterations
m.options.MAX_ITER = 5000

# Set number of nodes per time segment
m.options.NODES = 5

#solve the problem
m.solve(disp=True)

#plot
print("final val = " + str(tf.value[0]))
tm_adj = tm* tf.value[0]
plt.figure()
plt.plot(tm_adj[:-1], T_x.value[:-1], 'r')
plt.plot(tm_adj[:-1], T_y.value[:-1], 'b',)
plt.plot(tm_adj[:-1], T_z.value[:-1], 'k',)
plt.xlabel('Time [s]')
plt.ylabel('thrust [N]')

plt.savefig("Thrust_Vector_Components_v2.PNG")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x.value,y.value,z.value)
ax.set_xlabel("X Coord [m]")
ax.set_ylabel("Y Coord [m]")
ax.set_zlabel("Z Coord [m]")

plt.savefig("Trajectory_opt_LT.PNG")
plt.show()

