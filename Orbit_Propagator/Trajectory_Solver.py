"""
Trajectory_Solver Class
By Cameron Castillo
--------------------------
This script creates a class for the gekko solver used to find the
trajectory and thrust vector. It has several functions for finding 
the orbital maneuvers:
    
find_traj_opt_fuel:
Determine best trajectory which optimizes for fuel

find_traj_opt_time:
Determine best trajectory which optimizes for time

find_traj_gen:
find generic trajectory optimizing for both fuel and time with cost
values on each parameter
"""
import tools as t
import numpy as np
from gekko import GEKKO
from OrbitPropagator import OrbitPropagator as OP
import Planetary_Data as pd

class traj_solver:
    def __init__(self, c0, c1, init_pos, final_pos, max_height, 
                 n_steps, max_u, fuel_cost=1, time_cost=1, min_height=300,
                 num_orbits=10, dt=100, pass_coes=True, cb=pd.earth, 
                 remote_solve=False):
        #Set self parameters
        self.c0 = c0
        self.c1 = c1
        self.num_orbits = num_orbits
        self.dt=dt
        self.coes=pass_coes
        self.init_percent = init_pos
        self.final_percent = final_pos
        self.min_height = min_height
        self.max_height = max_height
        self.cb=cb
        self.n_steps = n_steps
        self.max_u = max_u
        self.fuel_cost = fuel_cost
        self.time_cost = time_cost
        self.remote_solve = remote_solve
        
        #solve for the initial orbits of both 
        self.find_init_orbit()
    
    def find_init_orbit(self):
        #global timespan
        tspan0 = t.find_period(self.c0[0])
        tspan1 = t.find_period(self.c1[0])
        if tspan0 > tspan1:
            self.tspan = self.num_orbits*tspan0
        else:
            self.tspan = self.num_orbits*tspan1

        self.op0 = OP(self.c0,tspan0,self.dt,self.coes)
        self.op1 = OP(self.c1,tspan1,self.dt,self.coes)

        self.start_index = int(len(self.op0.rs[:,0])*(self.init_percent/100))
        self.final_index = int(len(self.op1.rs[:,0])*(self.final_percent/100))

        self.max_orb = self.cb['radius'] + self.max_height
        self.min_orb = self.cb['radius'] + self.min_height
    
    def find_traj_gen(self):
        m = GEKKO(remote=self.remote_solve)

        #define time varible
        m.time = np.linspace(0, 1, self.n_steps)

        #define fixed variables
        theta_i = m.FV(lb = 0, ub = 360)
        theta_f = m.FV(lb = 0, ub = 360)

        rx_i = m.Var(self.op0.rs[self.start_index,0])
        ry_i = m.Var(self.op0.rs[self.start_index,1])
        rz_i = m.Var(self.op0.rs[self.start_index,2])

        vx_i = m.Var(self.op0.vs[self.start_index,0])
        vy_i = m.Var(self.op0.vs[self.start_index,1])
        vz_i = m.Var(self.op0.vs[self.start_index,2])

        rx_f = m.Var()
        ry_f = m.Var()
        rz_f = m.Var()

        vx_f = m.Var()
        vy_f = m.Var()
        vz_f = m.Var()

        #o1_tspan = m.Var()
        #o2_tspan = m.Var()

        ta_init = np.rad2deg(self.op0.ta)
        ta_final = np.rad2deg(self.op1.ta)

        #t_vect0 = np.linspace(0,tspan0,len(op0.rs))
        #t_vect1 = np.linspace(0,tspan2,len(op1.rs))

        m.cspline(theta_i, rx_i, ta_init, self.op0.rs[:, 0])
        m.cspline(theta_i, ry_i, ta_init, self.op0.rs[:, 1])
        m.cspline(theta_i, rz_i, ta_init, self.op0.rs[:, 2])

        m.cspline(theta_i, vx_i, ta_init, self.op0.vs[:, 0])
        m.cspline(theta_i, vy_i, ta_init, self.op0.vs[:, 1])
        m.cspline(theta_i, vz_i, ta_init, self.op0.vs[:, 2])

        m.cspline(theta_f, rx_f, ta_final, self.op1.rs[:, 0])
        m.cspline(theta_f, ry_f, ta_final, self.op1.rs[:, 1])
        m.cspline(theta_f, rz_f, ta_final, self.op1.rs[:, 2])

        m.cspline(theta_f, vx_f, ta_final, self.op1.vs[:, 0])
        m.cspline(theta_f, vy_f, ta_final, self.op1.vs[:, 1])
        m.cspline(theta_f, vz_f, ta_final, self.op1.vs[:, 2])

        r1 = m.Var(rx_i)
        r2 = m.Var(ry_i)
        r3 = m.Var(rz_i)

        r1dot = m.Var(vx_i)
        r2dot = m.Var(vy_i)
        r3dot = m.Var(vz_i)

        u1 = m.MV(lb = -self.max_u, ub = self.max_u)
        u1.STATUS = 1
        u2 = m.MV(lb = -self.max_u, ub = self.max_u)
        u2.STATUS = 1
        u3 = m.MV(lb = -self.max_u, ub = self.max_u)
        u3.STATUS = 1

        #define final time parameter
        tf = m.FV(value=1.0, lb=0.001, ub=self.tspan)
        tf.STATUS = 1

        m.Equation(r1.dt()/tf == r1dot)
        m.Equation(r2.dt()/tf == r2dot)
        m.Equation(r3.dt()/tf == r3dot)

        r = m.Var(lb=self.min_orb, ub=self.max_orb)

        m.Equation(r == m.sqrt(r1**2 + r2**2 + r3**2))
        #v = m.Intermediate(m.sqrt(r1dot**2 + r2dot**2 + r3dot**2))

        m.Equation(-self.cb['mu']*r1/r**3 == r1dot.dt()/tf + u1)
        m.Equation(-self.cb['mu']*r2/r**3 == r2dot.dt()/tf + u2)
        m.Equation(-self.cb['mu']*r3/r**3 == r3dot.dt()/tf + u3)

        m.Minimize(self.fuel_cost*m.integral(u1**2 + u2**2 + u3**2))
        m.Minimize(self.time_cost*tf)

        final = np.zeros(len(m.time))
        final[-1] = 1
        final = m.Param(value=final)

        m.Obj(final*(r1 - self.op1.rs[self.final_index, 0])**2)
        m.Obj(final*(r2 - self.op1.rs[self.final_index, 1])**2)
        m.Obj(final*(r3 - self.op1.rs[self.final_index, 2])**2)

        m.Obj(final*(r1dot - self.op1.vs[self.final_index, 0])**2)
        m.Obj(final*(r2dot - self.op1.vs[self.final_index, 1])**2)
        m.Obj(final*(r3dot - self.op1.vs[self.final_index, 2])**2)

        m.options.IMODE = 6
        m.options.solver = 3
        #m.options.ATOL = 1e-3
        m.options.MAX_ITER = 5000
        m.solve(disp=True)

        tm_adj = m.time * tf.VALUE[0]
        
        postion = np.array([[r1.VALUE], [r2.VALUE], [r3.VALUE]])
        thrust = np.array([[u1.VALUE], [u2.VALUE], [u3.VALUE]])

        return postion, thrust, tm_adj





