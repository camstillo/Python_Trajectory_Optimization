"""
Tools.py
"""
import numpy as np
import math as m

import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D

import Planetary_Data as pd

#degrees to rad
d2r = np.pi/180.0

def plot_n_orbits(rs, labels, cb=pd.earth, show_plot=False, save_plot=False, title='Trajectory Plot'):
    #Set dark background
    plt.style.use('dark_background')
    
    #define fig object
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111,projection='3d')
     
    # plot trajectory
    n = 0
    for r in rs:
        ax.plot(r[:,0],r[:,1],r[:,2],'b', label=labels[n])
        ax.plot(r[0,0], r[0,1], r[0,2], marker='o',
                 color='w', label=labels[n]+' Init Pos')
        n+=1
     
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
     
    max_val=np.max(np.abs(rs[0,0]))
     
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

def coes2rv(coes, deg=False, mu=pd.earth['mu']):
    if deg:
        # --------------------
        # NOTE: Add date parameter
        # when TLE functionality is
        # in added
        # --------------------
        a,e,i,ta,aop,raan = coes
        i *= d2r
        ta *= d2r
        aop *= d2r
        raan *= d2r
    else:
        a,e,i,ta,aop,raan = coes
        
    E = ecc_anomaly([ta,e], 'tae')
    
    r_norm = a*(1 - e**2)/(1  + e*np.cos(ta))
    
    # Calculate r and v vectors in perifocal frame
    r_perif = r_norm*np.array([m.cos(ta), m.sin(ta), 0])
    v_perif = m.sqrt(mu*a)/r_norm*np.array([-m.sin(E), m.cos(E)*m.sqrt(1 - e**2), 0])
    
    # rotation matrix from perifocal to ECI
    perif2eci = np.transpose(eci2perif(raan, aop, i))
    
    r = np.dot(perif2eci, r_perif)
    v = np.dot(perif2eci, v_perif)
    
    return r, v #, date

def eci2perif(raan, aop, i):
    row0 = [-m.sin(raan)*m.cos(i)*m.sin(aop) + m.cos(raan)*m.cos(aop),
            m.cos(raan)*m.cos(i)*m.sin(aop) + m.sin(raan)*m.cos(aop),
            m.sin(i)*m.sin(aop)]
    row1 = [-m.sin(raan)*m.cos(i)*m.cos(aop) - m.cos(raan)*m.sin(aop),
            m.cos(raan)*m.cos(i)*m.cos(aop) - m.sin(raan)*m.sin(aop),
            m.sin(i)*m.cos(aop)]
    row2 = [-m.sin(raan)*m.sin(i),
            -m.cos(raan)*m.sin(i),
            m.cos(i)]
    return np.array([row0,row1,row2])

def ecc_anomaly(arr, method, tol=1e-8):
    if method == 'newton':
        #Newton's method for iteratitavely finding E
        Me, e = arr
        if Me < np.pi/2.0: 
            E0 = Me + e/2.0
        else:
            E0 = Me - e
        
        E1 = 0
        for n in range(200): # arbitrary max n steps
            ratio = (E0 - e*np.sin(E0) - Me)/(1 - e*np.cos(E0))
            if abs(ratio) < tol:
                if n == 0: return E0
                else: return E1
            else:
                E1 = E0 - ratio
                E0 = E1
        # Did not converge!
        return False
    elif method == 'tae':
        ta, e = arr
        return 2*m.tan(m.sqrt((1-e)/(1+e))*m.tan(ta/2.0))
    else:
        print('invalid method for ecc anomaly')
        
def tle2coes(tle_filename, mu=pd.earth['mu']):
    # read tle file
    with open(tle_filename,'r') as f:
        lines = f.readlines()
    
    # separate into three lines
    line0 = lines[0].strip()    # satellite name
    line1 = lines[1].strip().split()
    line2 = lines[2].strip().split()
    
    # epoch (yr and day)
    epoch = line1[3]
    year, month, day, hour = calc_epoch(epoch)
    
    #collect coes
    
    # inclination
    i = float(line2[2])*d2r # rad
    # raan
    raan = float(line2[3])*d2r # rad
    # eccentricity
    e_string = line2[4]
    e = float('0.' + e_string) # rad
    # arg of perigee
    aop = float(line2[5])*d2r # rad
    # mean anomaly
    Me = float(line2[6])*d2r # rad
    # mean motion
    mean_motion = float(line2[7]) # revs / day
    # period
    T = 1/mean_motion*24*3600 # seconds
    #semi major axis
    a = (T**2*mu/4.0/np.pi**2)**(1/3.0)
    
    # calculate ecc anomaly
    E = ecc_anomaly([Me, e], 'newton')
    
    # calculate true anomaly
    ta = true_anomaly([E, e])
    
    return a,e,i,ta,aop,raan,[year, month, day, hour]
    
def calc_epoch(epoch):
    # epoch year
    year = int('20' + epoch[:2])
    
    epoch = epoch[2:].split('.')
    
    # day of year
    day_of_year = int(epoch[0]) - 1
    
    # decimal hr of day
    hour = float('0.' + epoch[1])*24.0
    
    # get year-month-day
    date = datetime.date(year,1,1) + datetime.timedelta(day_of_year)
    
    # extract month and day
    month = float(date.month)
    day = float(date.day)
    
    return year, month, day, hour

def true_anomaly(arr):
    E,e = arr
    return 2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E/2.0))

def tle2rv(tle_filename):
    return coes2rv(tle2coes(tle_filename))

def true_anomaly_r(rs, ta_init=0, mu=pd.earth['mu']):
    # unpack state
    rx, ry, rz, vx, vy, vz = rs
    
    # make vectors from rs
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    
    # flip velocity if necessary
    rdotv = np.dot(r,v)
    if rdotv < 0:
        v = 2*np.pi - v
    
    # define quantities for equation
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    #find eccentricity vector
    E_vect = ((v_mag**2/mu) - (1/r_mag))*r - ((rdotv)/mu)*v
    E_mag = np.linalg.norm(E_vect)
    
    #solve for total anomaly
    ta = np.arccos(np.dot(E_vect, r)/(E_mag*r_mag)) + ta_init
    
    return ta
    
def find_period(a, mu=pd.earth['mu']):
    return 2*np.pi*np.sqrt(a**3/mu)
    
    