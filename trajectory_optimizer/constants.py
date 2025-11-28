from math import pi

"""CONSTANTS"""
g_s = 3.73                       # gravitational acceleration (m/s^2)
rho_s = 0.020                    # surface density (kg/m^3)
beta_neg = 11.1*1000             # scale height (m)
rM = 3389500                     # radius of mars (m)
cD = 1.5                         # drag coefficient
Diam = 4.5                       # effective diameter (m)
S = pi*(Diam/2)**2               # planform area (m^2)
m_e = 3380                       # entry mass (kg)
h_e = 125000                     # entry height (m)
x_e = 0                          # entry downrange distance (m)
sb = 5.670374419 * 10**(-8)      # Stefan–Boltzmann constant (Wm^-2K^-4)
e_w = 0.89                       # emissivity
ballistic_coeff = m_e/(S*cD)     # ballistic coefficient (kg/m^2)
initial_time = 0.0               # initial trajectory time (s)
final_time = 500.0               # 

"""CONSTRAINT LIMITS"""
max_heat_flux = 225              # maximum allowable heat flux (W/cm^2)
max_Tw = 1873                    # maximum allowable TPS wall temp (K)
max_total_heat_load = 5477       # maximum allowable total heat laod (J/cm^2)
max_decel = 5                    # maximum allowable total Gs
