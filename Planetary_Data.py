# -*- coding: utf-8 -*-
"""
Planetary Data File

@author: Cameron
"""

G_meter = 6.67408e-11  # m
G = G_meter*10**-9     # km

sun = {
     'name':'Sun',
     'mass':1.989e30,   #kg
     'mu':1.32712e11,   #km
     'radius':695700.0  #km
     }

earth = {
    'name':'Earth',
    'mass':5.972e24,
    'mu':5.972e24*G,
    'radius':6378.0
    }

