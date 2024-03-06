
from gekko import GEKKO
import numpy as np
from skyfield.api import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize GEKKO Solver
m = GEKKO()

# Define time horizon and discretization
m.time = np.linspace(0, 2 * 365, 200) # day

# Obtain Earth position and velocity data
ts = load.timescale()
tx = ts.utc(2010, 1, 1)
planets = load('de421.bsp')
earthx = planets['earth']

# Define center body standard graviational parameter
mu = m.Const(2.959e-4) # au^3 / day^2

# Initialize simulation variables
x = m.Var(value=earthx.at(tx).position.au[0]) # au
y = m.Var(value=earthx.at(tx).position.au[1]) # au
z = m.Var(value=earthx.at(tx).position.au[2]) # au
vx = m.Var(value=(earthx.at(tx)).velocity.au_per_d[0]) # au / day
vy = m.Var(value=(earthx.at(tx)).velocity.au_per_d[1]) # au / day
vz = m.Var(value=(earthx.at(tx)).velocity.au_per_d[2]) # au / day

# Maximum Thrust to mass ratio (acceleration)
T_max = 2e-5 # au / day^2

# Thrust control variable
T = m.MV(lb=-T_max, ub=T_max)
T.STATUS = 1

# Define intermediate quantities
R = m.Intermediate((x**2 + y**2 + z**2)**0.5)
v = m.Intermediate((vx**2 + vy**2 + vz**2)**0.5)
ax = m.Intermediate(x * -mu / R**3 + T * vx / v)
ay = m.Intermediate(y * -mu / R**3 + T * vy / v)
az = m.Intermediate(z * -mu / R**3 + T * vz / v)

# Governing equations
m.Equations((vx.dt() == ax, vy.dt() == ay, vz.dt() == az))
m.Equations((x.dt() == vx, y.dt() == vy, z.dt() == vz))

# Integrated thrust (fuel usage)
J = m.Var(0)

# Equation relating thrust to fuel usage
m.Equation(J.dt() == m.abs2(T))

# Time mask for final time
final = np.zeros_like(m.time)
final[-1] = 1.0
final = m.Param(value=final)

# Constraint on final distance to center body
m.Equation(R * final < 0.2) # au

# Objective, minimizing fuel usage
m.Obj(J * final)

# Set solver mode
m.options.IMODE = 6

# Increase maximum number of allowed iterations
m.options.MAX_ITER = 2000

# Set number of nodes per time segment
m.options.NODES = 5

# Run solver and display progress
m.solve(disp=True)

# Plot orbit in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x.value, y.value, z.value)



# Plot Thrust history
plt.figure()
plt.plot(T.value[:-1])
plt.xlabel('Time [day]')
plt.ylabel('Thrust to Mass Ratio [AU / day^2]')

# Plot orbit distance history
plt.figure()
plt.plot(R.value)
plt.xlabel('Time [day]')
plt.ylabel('Distance to Center Body [AU]')

plt.savefig("Position_plot.PNG")
