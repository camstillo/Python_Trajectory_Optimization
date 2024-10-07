"""
Orbit propagator class
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

import Planetary_Data as pd
import tools

import ThrusterDict as td

def null_perts():
    return {
        'J2':False,
        'aero':False,
                #'Cd':2.2,           # Standard value, not necessarily constant
               #'A':(1e-3)**2/4.0   # 
               # },
        'moon_grav':False,
        'solar_grav':False,
        'thrust':False,
        'thrust_direction':1
    }

class OrbitPropagator:
    '''
    __init__
    class constructor fn, initializes state
    state0 initial state
    tspan simulation time span
    dt simulation time step 
    coes feed Keplerian orbital parameters (auto False)
    deg use degrees (True) or rads (False) (auto True)
    auto_orbit_prop automatically propagate orbit (auto True)
    cb central body determination (auto pd.earth)
    perts passes values for pert dict, determines which perturbations are used
    mass0 define mass of satellite, set to 1 kg for 1U Cubesat
    thruster identify thruster used by program
    thrust_function array of 1 or 0 which determines whether the thruster is on or not
    '''
    def __init__(self, state0, tspan, dt, coes=False, deg=True, 
                 auto_orbit_prop=True, cb=pd.earth, perts=null_perts, mass0=1.0, 
                 thruster=td.DummyThruster, ignore_fuel_mass=False, thrust_function=[]):
        if coes:
            self.r0, self.v0 = tools.coes2rv(state0, deg=deg, mu=cb['mu'])
        else:
            self.r0 = state0[:3]
            self.r0 = state0[3:]
        
        self.y0 = self.r0.tolist() + self.v0.tolist() + [mass0]
        self.tspan = tspan
        self.dt = dt
        self.cb = cb
        
        #find total num of steps
        self.n_steps = int(np.ceil(self.tspan/self.dt)) 
        
        #initialize arrays
        self.ys = np.zeros((self.n_steps,7)) 
        self.ts = np.zeros((self.n_steps, 1))
        self.mass = np.zeros((self.n_steps, 1))
        
        #initial conditions
        self.ts[0] = 0
        self.ys[0,:] = self.y0
        self.step = 1
        
        #check if thrust function is populated, otherwise it's all 1's
        #self.thrust_function = thrust_function
        #if not thrust_function.any():
        #    self.thrust_function = np.ones((self.n_steps, 1))
        #self.thrust_function_value = self.thrust_function[0][0]
        #print(self.thrust_function_value)
        
        #find initial true anomaly
        self.ta = np.zeros((self.n_steps, 1))
        self.ta[0] = tools.true_anomaly_r(self.y0)
        
        #initiate the solver
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator('lsoda')
        self.solver.set_initial_value(self.y0, 0) #, self.thrust_function_value)
        
        self.perts = perts
        self.mass0 = mass0 #don't get rid of this, necessary for mass finding function
        self.mass[0] = self.mass0
        self.ignore_fuel_mass = ignore_fuel_mass
        
        
            
        self.thruster = thruster
        
        #propagate the orbit if no other stimulus
        if auto_orbit_prop:
            self.propagate_orbit()
            
        
        
    def propagate_orbit(self):
        #propagate orbit
        while self.solver.successful and self.step < self.n_steps:
            self.solver.integrate(self.solver.t + self.dt)
            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y
            #self.thrust_function_value = self.thrust_function[self.step]
            self.ta[self.step] = tools.true_anomaly_r(self.ys[self.step])
            self.mass[self.step] = self.mass[self.step-1] + self.dt*self.ys[self.step,-1]
            self.step += 1
            
        #get all the radius values
        self.ts = self.ts[:self.step]
        self.rs = self.ys[:self.step,:3] 
        self.vs = self.ys[:self.step,3:6]
        
        self.alts = (np.linalg.norm(self.rs, axis=1) - self.cb['radius']).reshape((self.step,1))
        
    def diffy_q(self, t, y): #, thrust_enb):
        #unpack state
        rx, ry, rz, vx, vy, vz, mass = y
        
        #define radius and velocity array as np vector
        r=np.array([rx,ry,rz])
        v=np.array([vx,vy,vz])
        
        #norm of radius vector
        norm_r=np.linalg.norm(r)
        
        #define 2-body accel
        a =-r*self.cb['mu']/norm_r**3
        
        #Oblateness perturbation
        if self.perts['J2']:
            z2 = r[2]**2 
            r2 = norm_r**2 
            tx = r[0]/norm_r*(5*z2/r2 - 1)
            ty = r[1]/norm_r*(5*z2/r2 - 1)
            tz = r[2]/norm_r*(5*z2/r2 - 3)
            
            a_j2 = 1.5*self.cb['J2']*self.cb['mu']*self.cb['radius']**2/norm_r**4*np.array([tx,ty,tz])
            
            a += a_j2
        
        #Aerodynamic drag perturbation
        if self.perts['aero']:
            #Calc attitude & air density
            z = norm_r - self.cb['radius'] #altitude
            rho = tools.calc_atmospheric_density(z) #Air density at altitude
                        
            #Calc motion of s/c w/r/t rotating atmosphere
            v_rel = v - np.cross(self.cb['atm_rot_vector'],r)
            
            drag = -v_rel*0.5*rho*np.linalg.norm(v_rel)*self.perts['Cd']*self.perts['A']/mass
        
            a += drag
            
        if self.perts['thrust']:
            
            a_thrust = self.perts['thrust_direction']*tools.normed(v)*self.perts['thrust']/mass/1000.0
            #a_thrust *= thrust_enb #self.thrust_function_value
            
            #derivative of total mass
            dmdt = -self.perts['thrust']/self.perts['isp']/9.81
            #dmdt *= thrust_enb#self.thrust_function_value
            
            #check if ignore fuel mass is on and that all fuel has not been expended
            #Note: Fuel mass in grams, divide by 1000
            if((not self.ignore_fuel_mass) and ((self.mass0-mass) > self.thruster.FuelMass/1000.0)):
                a_thrust = 0
                dmdt = 0
            
            a += a_thrust
        
        #Pass derivatives of the state
        return [vx, vy, vz, a[0], a[1], a[2], dmdt]
    
    def calculate_coes(self):
        print("calculating coes...")
        
        self.coes = np.zeros((self.n_steps,6))
        
        for n in range (self.n_steps):
            self.coes[n,:] = tools.rv2coes(self.rs[n,:], self.vs[n,:], mu=self.cb['mu'], 
                                       degrees=True)
            
    def plot_coes(self, hours=False, days=False, show_plot=False, save_plot=False,
                  title="COEs", figsize=(16,8)):
        print('Plotting COEs')
        
        #create figures and axes instance
        fig,axs = plt.subplots(nrows=2,ncols=3,figsize=figsize)
        
        #figure title
        fig.suptitle(title,fontsize=20)
        
        if hours:
            ts = self.ts/3600
            x_unit = 'Hours'
        elif days:
            ts = self.ts/(3600.0*24.0)
            x_unit='Days'
        else:
            ts = self.ts
            x_unit = 'Seconds'
        
        #plot true anomaly
        axs[0,0].plot(ts,self.coes[:,3])
        axs[0,0].set_title('True Anomaly vs Time')
        axs[0,0].grid(True) 
        axs[0,0].set_ylabel('Angle [deg]')
        #axs[0,0].set_xlabel('Time [%s]' % x_unit)
        
        #plot semi-major axis
        axs[1,0].plot(ts,self.coes[:,0])
        axs[1,0].set_title('Semi-major axis vs Time')
        axs[1,0].grid(True) 
        axs[1,0].set_ylabel('Semi major axis [km]')
        axs[1,0].set_xlabel('Time [%s]' % x_unit)
        
        #plot eccentricity
        axs[0,1].plot(ts,self.coes[:,1])
        axs[0,1].set_title('Eccentricity vs Time')
        axs[0,1].grid(True) 
        #axs[0,1].set_xlabel('Time [%s]' % x_unit)
        
        #plot AOP
        axs[0,2].plot(ts,self.coes[:,4])
        axs[0,2].set_title('Argument of Periapse vs Time')
        axs[0,2].grid(True) 
        axs[0,2].set_ylabel('AOP [deg]')
        #axs[0,2].set_xlabel('Time [%s]' % x_unit)
        
        # plot inclination
        axs[1,1].plot(ts,self.coes[:,2])
        axs[1,1].set_title('Inclination vs Time')
        axs[1,1].grid(True) 
        axs[1,1].set_ylabel('inclination [deg]')
        axs[1,1].set_xlabel('Time [%s]' % x_unit)
        
        #plot RAAN
        axs[1,2].plot(ts,self.coes[:,5])
        axs[1,2].set_title('RAAN vs Time')
        axs[1,2].grid(True) 
        axs[1,2].set_ylabel('RAAN [deg]')
        axs[1,2].set_xlabel('Time [%s]' % x_unit)
        
        if show_plot:
            plt.show()
        
        if save_plot:
            plt.savefig(title+'.png',dpi=800)
    
    def plot_3d(self, show_plot=False, save_plot=False, title='Trajectory Plot'):
        #Set dark background
        plt.style.use('dark_background')
        
        #define fig object
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111,projection='3d')
        ax.view_init(elev=45, azim=-90)
        
        # plot trajectory
        ax.plot(self.rs[:,0],self.rs[:,1],self.rs[:,2],'b', label='Trajectory')
        ax.plot(self.rs[0,0], self.rs[0,1], self.rs[0,2], marker='o',
                color='w', label='Initial Position')
        
        # plot central body
        # _u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
        # _x = self.cb['radius']*np.cos(_u)*np.sin(_v)
        # _y = self.cb['radius']*np.sin(_u)*np.sin(_v)
        # _z = self.cb['radius']*np.cos(_v)
        # ax.plot_surface(_x, _y, _z, cmap='Blues')
        
        #plot the x, y, z vectors
        l = self.cb['radius'] * 2
        x,y,z=[[0,0,0],[0,0,0],[0,0,0]]
        u,v,w=[[l,0,0],[0,l,0],[0,0,l]]
        ax.quiver(x,y,z,u,v,w,color='k')
        
        max_val=np.max(np.abs(self.rs))
        
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        
        ax.set_xlabel(['X (km)'])
        ax.set_ylabel(['Y (km)'])
        ax.set_zlabel(['Z (km)'])
        
        ax.set_aspect('equal')
        
        ax.set_title(title)
        plt.legend()
        
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png', dpi=300)
            
    def plot_alts (self, show_plot=False, save_plot=False, hours=False, days=False,
                   title='Radial Distance vs Time', figsize=(16,8), dpi=500):
        if hours:
            ts = self.ts/3600
            x_unit = 'Hours'
        elif days:
            ts = self.ts/(3600.0*24.0)
            x_unit='Days'
        else:
            ts = self.ts
            x_unit = 'Seconds'
        
        plt.figure(figsize=figsize)
        plt.plot(ts, self.alts, 'w')
        plt.grid(True)
        plt.xlabel('Time [%s]' % x_unit)
        plt.ylabel('Altitude [km]')
        plt.title(title)
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=dpi)
            
    def calc_apoapse_periapse (self):
        self.apoapses = self.coes[:,0]*(1 + self.coes[:,1])
        self.periapses = self.coes[:,0]*(1 - self.coes[:,1])
        
    def plot_apoapse_periapse(self, hours=False, days=False, show_plot=False, 
                              save_plot=False, title='Apoapse and Periapse vs Time',
                              dpi=500):
        plt.figure(figsize=(20,10))
        
        if hours:
            ts = self.ts/3600
            x_unit = 'Hours'
        elif days:
            ts = self.ts/(3600.0*24.0)
            x_unit='Days'
        else:
            ts = self.ts
            x_unit = 'Seconds'
            
        #plot each
        plt.plot(ts, self.apoapses, 'b', label='Apoapse')
        plt.plot(ts, self.periapses, 'r', label='Periapses')
        plt.xlabel('Time [%s]' % x_unit)
        plt.ylabel('Altitude [km]')
        
        plt.grid(True)
        plt.title(title)
        plt.legend()
        
        if show_plot:
            plt.show()
        else:
            plt.savefig(title+'.png',dpi=dpi)
            
    def plot_mass(self, hours=False, days=False, show_plot=False, 
                              save_plot=False, title='Satellite Mass vs Time',
                              dpi=500):
        plt.figure(figsize=(20,10))
        
        if hours:
            ts = self.ts/3600
            x_unit = 'Hours'
        elif days:
            ts = self.ts/(3600.0*24.0)
            x_unit='Days'
        else:
            ts = self.ts
            x_unit = 'Seconds'
            
        #plot Mass
        plt.plot(ts, self.mass, 'b', label='Apoapse')
        plt.xlabel('Time [%s]' % x_unit)
        plt.ylabel('Mass [kg]')
        
        plt.grid(True)
        plt.title(title)
        plt.legend()
        
        if show_plot:
            plt.show()
        else:
            plt.savefig(title+'.png',dpi=dpi)
        
