# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:59:41 2024

@author: ASGRM
"""

from math import cos, sin


# Function more native to the embedded implementation without complex numbers.
def R_Ph_E_emb(U:float, IL:float, IE:float, theta_U:float, theta_IL:float, theta_IE:float, RE_RL:float, XE_XL:float)-> float: 
    '''
    Function to calculate the resistive part of the phase-earth loop impedance.
    The function do not use complex numbers. 
    Reference: Gerhard Ziegler - "Numerical distance protection" page 104 (3-51)

    Parameters
    ----------
    U : float
        Magnitude of phase-earth voltage [V].
    IL : float
        Magnitude of phase fault current [A].
    IE : float
        Magnitude of earth fault current [A].
    theta_U : float
        Phase angle of the phase-earth voltage U in radians..
    theta_IL : float
        Phase angle of the phase fault current IL in radians..
    theta_IE : float
        Phase angle of the earth fault current IE in radians.
    RE_RL : float
        Resistive residual compensation factor RE_RL = (1/3) * (R0/R1 - 1).
    XE_XL : float
        Reactive residual compensation factor XE_XL = (1/3) * (X0/X1 - 1).

    Returns
    -------
    float
        Real/resistive part of the phase-earth impedance loop [Ohm].

    '''
    Rph_E = (U/IL) * ( (cos(theta_U - theta_IL) - IE/IL*XE_XL*cos(theta_U - theta_IE)) \
                     / (1 - (XE_XL + RE_RL)*IE/IL*cos(theta_IE - theta_IL) + RE_RL*XE_XL*(IE/IL)**2) )
    
    return Rph_E

def X_Ph_E_emb(U:float, IL:float, IE:float, theta_U:float, theta_IL:float, theta_IE:float, RE_RL:float, XE_XL:float)-> float: 
    '''
    Function to calculate the reactive part of the phase-earth loop impedance.
    The function do not use complex numbers. 
    Reference: Gerhard Ziegler - "Numerical distance protection" page 104 (3-52)

    Parameters
    ----------
    U : float
        Magnitude of phase-earth voltage [V].
    IL : float
        Magnitude of phase fault current [A].
    IE : float
        Magnitude of earth fault current [A].
    theta_U : float
        Phase angle of the phase-earth voltage U in radians..
    theta_IL : float
        Phase angle of the phase fault current IL in radians..
    theta_IE : float
        Phase angle of the earth fault current IE in radians.
    RE_RL : float
        Resistive residual compensation factor RE_RL = (1/3) * (R0/R1 - 1).
    XE_XL : float
        Reactive residual compensation factor XE_XL = (1/3) * (X0/X1 - 1).

    Returns
    -------
    float
        Imaginary/reactive part of the phase-earth impedance loop [Ohm].

    '''
    Xph_E = (U/IL) * ( (sin(theta_U - theta_IL) - IE/IL*RE_RL*sin(theta_U - theta_IE)) \
                     / (1 - (XE_XL + RE_RL)*IE/IL*cos(theta_IE - theta_IL) + RE_RL*XE_XL*(IE/IL)**2) )
    return Xph_E