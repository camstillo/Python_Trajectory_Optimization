'''
Solved cart mass problem
Cameron Castillo
'''
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

#initialize solver
m = GEKKO()

#initial conditions
L           = 0.1       # m
theta_0     = 5*np.pi/6 # rads
cart_mass   = 1         # kg
bob_mass    = 0.3       # kg
drag_coef   = 0.01      # kg/s
force_max   = 300       # N
x_0         = 0         # m
grav        = 9.81      # m/s**2
final_angle = np.pi

#define time for solver
num_points = 101
tm = np.linspace(0,1,num_points)
m.time = tm # sec

#define constants and vars from the problem
length  = m.Const(value=L)
M_cart  = m.Const(value=cart_mass)
m_bob   = m.Const(value=bob_mass)
c_drag  = m.Const(value=drag_coef)
theta   = m.Var(value=theta_0, lb=0, ub=np.pi*2)
x       = m.Var(value=x_0)
dtheta  = m.Var()
dx      = m.Var()
ddtheta = m.Var()
ddx     = m.Var()

#define manipulated variables
F = m.MV(lb = -1*force_max, ub = force_max)
F.STATUS = 1

#define final time parameter
tf = m.FV(value=1.0, lb=0.001, ub=5000)
tf.STATUS = 1

#define system dynamics
m.Equations((dx==x.dt()/tf, dtheta==theta.dt()/tf))
m.Equations((ddx==dx.dt()/tf, ddtheta==dtheta.dt()/tf))
m.Equation(ddx/tf == (m_bob*length*(dtheta)**2*m.sin(theta) + 
            (m_bob*grav*m.sin(theta)*m.cos(theta)) -
            c_drag*dx + F)/
           (M_cart + m_bob - m_bob*(m.cos(theta))**2))
m.Equation(ddtheta/tf == ((-1*grav/length)*m.sin(theta)) - 
           (m.cos(theta)/length)*((m_bob*length*(dtheta)**2*m.sin(theta) +
                                 (m_bob*grav)*m.sin(theta)*m.cos(theta) +
                                 -1*c_drag*dx + F)/
                                (M_cart + m_bob - m_bob*(m.cos(theta))**2)))

#define final value
m.fix_final(var=theta, val=np.pi)

#define objective fn
m.Obj(x)

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
plt.plot(tm_adj[:-1], F.value[:-1])
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')

plt.savefig("Force_vs_time.PNG")

plt.figure()
plt.plot(tm_adj[:-1], x.value[:-1])
plt.xlabel('Time [s]')
plt.ylabel('x Position [m]')

plt.savefig("Pos_vs_time.PNG")

plt.figure()
plt.plot(tm_adj[:-1], theta.value[:-1])
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')

plt.savefig("Angle_vs_time.PNG")
  
