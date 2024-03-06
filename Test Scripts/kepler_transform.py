# -*- coding: utf-8 -*-
"""
Functions to transform Keplerian parameters to Cartesian Coordinates
"""
import numpy as np

'''
kep_to_cart
Turn Keplerian initial conditions into cartesian point. Returns the
initial cartesian position and velocity vectors.

Arguments:
a = semimajor axis
e = eccentricity
i = inclinations
w = arg of perigee
T = epoch
mu = graviational parameter for body = 3.986e14 at Earth

'''
def kep_to_cart(a, e, i, w, Ohm, T, mu=3.986e14):
    
    #Calculate Mean anomaly
    n = np.sqrt(mu/a**3)
    m = 
    return 0
