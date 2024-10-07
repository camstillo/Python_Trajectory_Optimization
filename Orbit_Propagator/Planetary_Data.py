# -*- coding: utf-8 -*-
"""
Planetary Data File

@author: Cameron
"""

from numpy import array



G_meter = 6.67408e-11  # m
G = G_meter*10**-9     # km

sun = {
     'name':'Sun',
     'mass':1.989e30,   #kg
     'mu':1.32712e11,   #km
     'radius':695700.0  #km
     }

#1973 atmospheric model, a close approximation
atm=array([[63.096, 2.059e-4],[251.189,5.909e-11],[1000.0,3.561e-15]])
earth = {
    'name':'Earth',
    'mass':5.972e24,
    'mu':5.972e24*G,
    'radius':6378.0,
    'J2':-1.082635854e-3,
    'zs':atm[:,0],  # km
    'rhos':atm[:,1]*10**9, #kg /km**3
    'atm_rot_vector':array([0.0,0.0,72.9211e-6])
    }

