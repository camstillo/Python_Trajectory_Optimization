"""
Solar Sail Dynamics Optimization
Cameron Castillo
"""
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

#start the solver
m = GEKKO()

#define constants
year_to_sec_conv = 365*24*3600   # s / yr
day_to_sec_conv  = 24*3600       # s / day
h_0       = 500e3                # initial orbital height (m)
h_f       = 600e3                # final orbital height (m)
r_earth   = 6.371e6              # earth radius (m)
r_0       = h_0 + r_earth        # initial radius (m)
v_tan_0   = 7.5969e3             # tangential velocty intial (m/s)
a_so      = 5.93e-3              # gravitational acc at Earth distance from sun (m/s)
h_min     = 200                  # min height (m)
light_num = 1.15e-1              # lightness number of sail (unitless)
mu_earth  = 3.986e14             # gravitational parameter earth (m**3/s**2)

#define initial conditions
mu                  = m.Const(value=mu_earth) 
sail_lambda         = m.Const(value=light_num) 
sail_acc            = m.Const(value=a_so)
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

#define time variable
num_points = 101
tm = np.linspace(0,1,num_points)
m.time = tm # sec

#define control angles (manip. vars)
beta = m.MV(value=0, lb=-np.pi/2, ub=np.pi/2)
beta.STATUS = 1
alpha = m.MV(value=np.pi/2, lb=0, ub=np.pi)
alpha.STATUS = 1

#define fixed variables
tf = m.FV(value=1.0, lb=0.001, ub=10*year_to_sec_conv)
tf.STATUS = 1

#define variable relationships
m.Equations((dx==x.dt()/tf, dy==y.dt()/tf, dz==z.dt()/tf))
m.Equations((ddx==dx.dt()/tf, ddy==dy.dt()/tf, ddz==dz.dt()/tf))
m.Equation(rad == m.sqrt(x**2 + y**2 + z**2))

#define system dynamics
m.Equation(ddx == (-1*mu/rad**3)*x + 
           sail_lambda*sail_acc*(m.sin(alpha)*m.cos(beta))**2
           *(m.sin(alpha)*m.cos(beta)))
m.Equation(ddy == (-1*mu/rad**3)*y + 
           sail_lambda*sail_acc*(m.sin(alpha)*m.cos(beta))**2
           *(m.sin(alpha)*m.sin(beta)))
m.Equation(ddy == (-1*mu/rad**3)*z + 
           sail_lambda*sail_acc*(m.sin(alpha)*m.cos(beta))**2
           *(m.cos(alpha)))

#define final value
m.fix_final(rad, val=h_f)

#define objective fn
m.Obj(beta)

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
plt.plot(tm_adj[:-1], beta.value[:-1], 'r')
plt.plot(tm_adj[:-1], alpha.value[:-1], 'b',)
plt.xlabel('Time [s]')
plt.ylabel('angle [rad]')

plt.savefig("Angle_vs_Tim_SS.PNG")

plt.figure()
plt.plot(tm_adj[:-1], dx.value[:-1], 'r')
plt.plot(tm_adj[:-1], dy.value[:-1], 'b')
plt.plot(tm_adj[:-1], dz.value[:-1], 'g')
plt.xlabel('Time [s]')
plt.ylabel('angle [rad]')

plt.savefig("Angle_vs_Time_SS.PNG")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x.value,y.value,z.value)
ax.set_xlabel("X Coord [m]")
ax.set_ylabel("Y Coord [m]")
ax.set_zlabel("Z Coord [m]")

plt.savefig("Trajectory_opt_SS.PNG")
#plt.show()




