from gekko import GEKKO
import numpy as np
import math
# from skyfield.api import load
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Make a function for spherical to cartesian
def sph_to_cart (r,theta,phi):
    x = r*math.sin(theta)*math.cos(phi)
    y = r*math.sin(theta)*math.sin(phi)
    z = r*math.cos(theta)
    return x,y,z

# Initialize GEKKO Solver
m = GEKKO()

# Define time horizon and discretization
year_to_sec_conv = 365*24*3600
# max_time = 10 * year_to_sec_conv # seconds
num_points = 50
m.time = np.linspace(0,1,num_points) # sec

# Obtain Earth position and velocity data
#ts = load.timescale()
#tx = ts.utc(2024, 1, 1)
#planets = load('de421.bsp')
#earthx = planets['earth']

# Define center body standard graviational parameter
mu = m.Const(3.986e14) # m^3 / s^2
lightness_lambda = m.Const(1.15e-1) # unitless
h_0      = 500e3         # initial orbital height (m)
r_earth  = 6.371e6       # earth radius (m)
r_0      = h_0 + r_earth # initial radius (m)
v_tan    = 7.5969e3      # tangential velocty intial (m/s)
a_uo     = 5.93e-3       # gravitational acc at Earth distance from sun (m/s)

# Initialize simulation variables
# all in spherical coordinates
x        = m.Var(value=r_0)         # m
y        = m.Var(value=0)           # m        
z        = m.Var(value=0)           # m
v_x      = m.Var(value=0)           # m / sec,   d(x)/dt
v_y      = m.Var(value=0)           # m / sec, d(y)/dt
v_z      = m.Var(value=v_tan)       # m / sec, d(z)/dt
rad      = m.Var(value=r_0)         # m, different from radius!
alpha    = m.Var(value=np.pi/2)

# Angle control variable
beta     = m.MV(lb = -np.pi, ub = np.pi)
beta.STATUS = 1
alpha    = m.MV(lb = 0, ub = np.pi)
alpha.STATUS = 1

# final time variables
tf = m.FV(value=1.0, lb=0.001, ub=10*year_to_sec_conv)
tf.STATUS = 1

#Solve for the radius
m.Equation(rad == m.sqrt(x**2 + y**2 + z**2))

# Define intermediate quantities
# NOTE: Possibly change to equation?
a_srp   = m.Intermediate(lightness_lambda*a_uo*(m.cos(beta)*m.sin(alpha))**2)
# radius  = m.Intermediate(m.sqrt(x**2 + y**2 + z**3))
a_grav  = m.Intermediate(-1*mu*rad**-3)
a_x     = m.Intermediate((a_grav*x) + a_srp*(m.sin(alpha)*m.cos(beta)))
a_y     = m.Intermediate((a_grav*y) + a_srp*(m.sin(alpha)*m.sin(beta)))
a_z     = m.Intermediate((a_grav*z) + a_srp*m.cos(alpha))


# Governing equations
# divide each governing equation derivative by the time to perform time scaling
m.Equations((v_x.dt()/tf == a_x, v_y.dt()/tf == a_y, v_z.dt()/tf == a_z))
m.Equations((x.dt()/tf == v_x, y.dt()/tf == v_y, z.dt()/tf == v_z))

#constraint on orbital height
h_min = 200e3 # de-orbit radius
m.Equation(rad > r_earth + h_min) # m

# Time mask for final time
#final = np.zeros_like(m.time)
#final[-1] = 1.0
#final = m.Param(value=final)

# Constraint on final distance to center body
h_final = 600e3 # final height (m)
m.fix_final(rad, val=h_final+r_earth)

# Objective, minimizing final time
m.Obj(tf)

# Set solver mode
m.options.IMODE = 6

# Increase maximum number of allowed iterations
m.options.MAX_ITER = 5000

# Set number of nodes per time segment
m.options.NODES = 5

# Run solver and display progress
m.solve(disp=True)

# Plot orbit in 3D
#[x,y,z] = sph_to_cart(radius.value,theta.value,phi.value)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x.value,y.value,z.value)
plt.xlabel("X Coord [m]")
plt.ylabel("Y Coord [m]")
plt.zlabel("Z Coord [m]")

plt.savefig("optimized_trajectory.PNG")

# Plot Thrust history
m.time.value = m.time.value*tf[0]
plt.figure()
plt.plot(beta.value[:-1], m.time.value[:-1])
plt.xlabel('Time [yr]')
plt.ylabel('beta angle [rad]')

plt.savefig("beta_angle.PNG")

# Plot orbit distance history
plt.figure()
# plt.plot(radius.value)
plt.xlabel('Time [s]')
plt.ylabel('Distance to Center Body [m]')

plt.savefig("radius_plot.PNG")
plt.show()
