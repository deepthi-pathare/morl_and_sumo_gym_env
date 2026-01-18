import math
import scipy.constants as sc

m = 44000 # Mass of the vehicle in Kg
Cd = 0.6 # Airdrag
Af = 10 # Frontal area in m2
Air_rho = 1.2 # Air density in kg/m3
Cr = 0.006 # Rolling resistance coefficient
g = sc.g # Acceleration of gravity in m/s2

def calculate_energy_consumed(a, v, slope, duration):
    """
       Calculate the energy consumed by ego vehicle
    Args:
        a (float): Acceleration in m/s2
        v (float): Velocity in m/s
        slope (float): Slope of the current vehicle position in degrees
    """       
    a = max(a, 0) # all negative acceleration are performed by service brakes. Regenration of Bevs neglatected.
    F = (m*a) + (0.5 * Cd * Af * Air_rho * v * v) + (g * Cr * m) + (m * g * math.sin(math.atan(slope/100)))
    P = F * v
    energy_consumed = (P/1000) * (duration/3600) # kwh
    return energy_consumed

def calculate_init_kinetic_energy(u):
    init_e = (0.5 * m * u * u) / (1000 * 3600)
    return init_e