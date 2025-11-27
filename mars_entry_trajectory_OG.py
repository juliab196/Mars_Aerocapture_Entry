#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:37:47 2024

@author: juliabaird
"""

import matplotlib.pyplot as plt
from math import sin, cos, tan, exp, pi, sqrt, asin
import numpy as np

#------------------------------------------------------------------------------
"""Define Parameters"""

g_s = 3.73                       # gravitational acceleration (m/s*^)
rho_s = 0.020                    # surface density (kg/m^3)
beta_neg = 11.1*1000             # scale height (m)
rM = 3389500                     # radius of mars (m)
cD = 1.5                         # drag coefficient
cL = 0.4                         # lift coefficient
Diam = 4.5                       # effective diameter (m)
S = pi*(Diam/2)**2               # planform area (m^2)
m_e = 3380                       # entry mass (kg)
v_e = 6000                       # entry velocity (m/s)
gamma_e = -11.5*pi/180           # entry flight angle
h_e = 125000                     # entry height (m)
x_e = 0                          # entry downrange distance (m)
sb = 5.670374419 * 10**(-8)      # Stefan–Boltzmann constant (Wm^-2K^-4)
e_w = 0.89                       # emissivity
ballistic_coeff = m_e/(S*cD)     # ballistic coefficient (kg/m^2)

#------------------------------------------------------------------------------
"""Flight Trajectory Differential Equations"""

def f(t, y, m_e, g_s, rM, beta_neg, cL, cD):
    
    # define v, gamma, x and h
    v = y[0]
    gamma = y[1]
    h = y[3]
    
    # calculate g and rho and r
    g = g_s*(rM/(rM+h))** 2 # local gravitational acceleration (m/s^2)
    rho = rho_s*exp(-h/beta_neg) # local air density (kg/m^3)
    r = rM + h
    # calculate lift and drag
    Lift = (1/2)*rho*v**2*S*cL
    Drag = (1/2)*rho*v**2*S*cD
    
    # calculate dv_dt
    dv_dt = -Drag/m_e-g*sin(gamma)
    
    # calculate d_gamma_dt
    dgamma_dt = (1/v)*(Lift/m_e - (g-v**2/r))*cos(gamma)

    # calculate dx_dt
    dx_dt = (rM/r)*v*cos(gamma)
    
    # calculate dh_dt
    dh_dt = v*sin(gamma)

    
    dy_dt = [dv_dt, dgamma_dt, dx_dt, dh_dt]
    return dy_dt

#------------------------------------------------------------------------------

"""Perform Numerical Integration"""

import time
from scipy.integrate import solve_ivp

initial_time = 0.0 # s
final_time = 500.0 # s
initial_conditions = [v_e, gamma_e, x_e, h_e] 
steps_to_evaluate = 10000
dt = final_time/steps_to_evaluate
rtol = 1.0e-6
atol = 1.0e-6
start_time = time.perf_counter()

solution = solve_ivp(lambda t, y: f(t, y, m_e, g_s, rM, beta_neg, cL, cD),
                     t_span = [initial_time, final_time], 
                     y0 = initial_conditions, 
                     method='RK45', 
                     dense_output=True, 
                     rtol = rtol, atol = atol, 
                     t_eval = np.linspace(initial_time, final_time, steps_to_evaluate))
end_time = time.perf_counter()

#------------------------------------------------------------------------------
"""Post Processing"""

# Extract results from numerical integration
time_results = solution.t
velocity_results = solution.y[0]
raw_gamma_results = solution.y[1]
down_range_results = solution.y[2]
altitude_results = solution.y[3]

# Determine secondary results  
gamma_results = []
rho_results = []
q_conv_results = []
q_rad_results = []
q_stag_total_results = []
acceleration_results = []
Tw_results = []
deceleration_results = []
C = 2.35*10**4; a = 0.525; b = 1.19 # Tauber-Sutton correation coefficients

for i in range(0,len(time_results)):
    # Caculate flight angle
    gamma_results.append((raw_gamma_results[i])*180/pi)
    
    # Calculate stagnation point heat flux
    rho = rho_s*exp(-altitude_results[i]/beta_neg)
    rho_results.append(rho)
    q_conv = 7.2074*(rho_results[i]**0.4739)*((Diam/2)**
            (-0.5405))*((velocity_results[i]/1000)**3.4956)
    q_conv_results.append(q_conv)
    if velocity_results[i] >= 6000:
        func_v = 0.2
    else:
        func_v = 0
    q_rad = C*((Diam/2)**a)*(rho**b)*func_v
    q_rad_results.append(q_rad)
    q_stag_total = q_rad + q_conv
    q_stag_total_results.append(q_stag_total)
    
    # Calculate stagnation point temperature
    Tw = ((q_stag_total)*10000/(sb*e_w))**(1/4)
    Tw_results.append(Tw)
    
    # Calculate deceleration 
    Lift = (1/2)*rho*(velocity_results[i]**2)*S*cL
    Drag = (1/2)*rho*(velocity_results[i]**2)*S*cD
    deceleration = ((1/2)*rho*(velocity_results[i]**2)*
                    ((1+(Lift/Drag)**2)**(1/2))*(S*cD/m_e))/9.81
    deceleration_results.append(deceleration)
  
  
# Determine local heat flux and local wall temperature
shell_angle = 70*pi/180
shell_h = (Diam/2)/tan(shell_angle)
x_length = (Diam/2)/sin(shell_angle)
x_segments = 100
x_list = []
alpha_list = []
local_heat_flux_list = []
local_Tw_list = []

for i in range(0,x_segments):
    # Find distance from stagnation point
    x = x_length*i/x_segments
    x_list.append(x)
    
    # Find body angle from stagnation point
    side_length = sqrt(shell_h**2 + x**2 - 2*shell_h*x*cos(shell_angle))
    alpha = asin(x*sin(shell_angle)/side_length)
    alpha_list.append(alpha)
    
for i in range(0,len(time_results)):
    local_heat_flux_at_i = []
    local_Tw_at_i = []
    for j in range(1,x_segments):
        # Find local heating across all x for a particular time step
        local_heat_flux = (9.43*10**(-5))*sqrt((rho_results[i]*cos(alpha_list[j])
                 *sin(alpha_list[j]))/x_list[j])*velocity_results[i]**3 # W/m^2
        local_heat_flux_at_i.append(local_heat_flux/10000)
        
        # Find local wall tempearture across all x for a particular time step
        local_Tw = (local_heat_flux/(sb*e_w))**(1/4) #K
        local_Tw_at_i.append(local_Tw)
    local_heat_flux_list.append(local_heat_flux_at_i)
    local_Tw_list.append(local_Tw_at_i)

# Determine total heat load across entire trajectory
heat_load_results = [0]*x_segments
for i in range(0,len(time_results)):
    for j in range(0,x_segments-1):
        heat_load_results[j] += local_heat_flux_list[i][j]*dt #J/cm^2
  
#------------------------------------------------------------------------------
"""Plot Results"""

# Trajectory plot
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(time_results, altitude_results/1000.0)
ax[0, 0].set_title('Time vs Altitude')
ax[0, 0].set_xlabel('Time (sec)')
ax[0, 0].set_ylabel('Altitude (km)')
ax[0, 1].plot(time_results, gamma_results, 'tab:orange')
ax[0, 1].set_title('Time vs Flight Angle')
ax[0, 1].set_xlabel('Time (sec)')
ax[0, 1].set_ylabel('Flight Angle (deg)')
ax[1, 0].plot(time_results, down_range_results/1000.0, 'tab:green')
ax[1, 0].set_title('Time vs Down range distance')
ax[1, 0].set_xlabel('Time (sec)')
ax[1, 0].set_ylabel('Down range distance (km)')
ax[1, 1].plot(time_results, velocity_results/1000, 'tab:red')
ax[1, 1].set_title('Time vs velocity')
ax[1, 1].set_xlabel('Time (sec)')
ax[1, 1].set_ylabel('Velocity (km/s)')
plt.tight_layout()
figure_filename = 'Mars Aerocapture entry'
plt.savefig(figure_filename + '.png')
plt.show()

# Convective and radiative heating plot
fig, ax = plt.subplots(1, 2)

ax[0].plot(time_results, q_conv_results,'tab:green')
ax[0].set_title('Time vs S.P. Convective Heat Flux')
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Convective Heat Flux (W/cm^2)')
ax[1].plot(time_results, q_rad_results, 'tab:orange')
ax[1].set_title('Time vs S.P. Radiative Heat Flux')
ax[1].set_xlabel('Time (sec)')
ax[1].set_ylabel('Radiative Heat Flux (W/cm^2)')
plt.tight_layout()
figure_filename = 'Convective and radiative heating'
plt.savefig(figure_filename + '.png')
plt.show()

# Velocity vs altitude plot
fig, ax = plt.subplots()
ax.plot(velocity_results/1000, altitude_results/1000.0, 'tab:blue')
ax.set_title('Velocity vs Altitude')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Altitude (km)')
plt.tight_layout()
figure_filename = 'Velocity vs Altitude'
plt.savefig(figure_filename + '.png')
plt.show()

# Deceleration plot
fig, ax = plt.subplots()
ax.plot(time_results, deceleration_results)
ax.legend
ax.set_title('Time vs Deceleration')
ax.set_xlabel('Time (sec)')
ax.set_ylabel("G's")
plt.tight_layout()
figure_filename = 'Deceleration'
plt.savefig(figure_filename + '.png')
plt.show()

# Stagnation point heat flux and wall temperature plot
fig, ax = plt.subplots(1,2)

ax[0].plot(time_results, q_stag_total_results, 'tab:purple')
ax[0].set_title('Time vs S.P. Heat Flux')
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Heat Flux (W/cm^2)')
ax[1].plot(time_results, Tw_results, 'tab:red')
ax[1].set_title('Time vs S.P. Wall Temperature ')
ax[1].set_xlabel('Time (sec)')
ax[1].set_ylabel('Wall Temperature (K)')
plt.tight_layout()
figure_filename = 'Stagnation point heat flux and temperature'
plt.savefig(figure_filename + '.png')
plt.show()

# Local Heat flux and wall temperataure plot
index_of_desired_time = 2000

fig, ax = plt.subplots(1,2)
ax[0].plot(x_list[1:],local_heat_flux_list[index_of_desired_time],'tab:purple' )
ax[0].set_title(f'Local Heat Flux at {time_results[index_of_desired_time]:.1f}s')
ax[0].set_xlabel('Distance from stagnation point (m)')
ax[0].set_ylabel('Heat Flux (W/cm^2)')
ax[1].plot(x_list[1:],local_Tw_list[index_of_desired_time],'tab:red' )
ax[1].set_title(f'Local Wall Temp. at {time_results[index_of_desired_time]:.1f}s')
ax[1].set_xlabel('Distance from stagnation point (m)')
ax[1].set_ylabel('Temperature (K)')
plt.tight_layout()
figure_filename = 'Local heat flux and local wall temperature '
plt.savefig(figure_filename + '.png')
plt.show()
  
# Total heat load plot
fig, ax = plt.subplots()
ax.plot(x_list, heat_load_results, 'tab:red')
ax.legend
ax.set_title('Total Heat Load')
ax.set_xlabel('Distance from stagnation point (m)')
ax.set_ylabel("Total Heat Load (J/cm^2)")
plt.tight_layout()
figure_filename = 'Heat load'
plt.savefig(figure_filename + '.png')
plt.show()

#------------------------------------------------------------------------------
"""Extract Key Parameters"""

# Key trajectory parameters
print('   Key trajectory parameters:')
print(f'Minimum altitude = {min(altitude_results)/1000:.2f} km')
index_of_min_alt = np.argmin(altitude_results)
print(f'Time of minimum altitude = {time_results[index_of_min_alt]:.2f} s')
print(f'Flight angle at minimum altitude = {gamma_results[index_of_min_alt]:.2f} deg')
print(f'Velocity at at minimum altitude = {velocity_results[index_of_min_alt]/1000:.2f} km/s')
print(f'Finishing altitude = {altitude_results[-1]/1000:.2f} km')
print(f'Maximum flight angle = {max(gamma_results):.2f} deg')
index_of_max_gamma = gamma_results.index(max(gamma_results))
print(f'Time of maximum flight angle = {time_results[index_of_max_gamma]:.2f} s')
print(f'Finishing flight angle = {gamma_results[-1]:.2f} deg')
print(f'Maximum down range distance = {max(down_range_results)/1000:.2f} km')
print(f'Minimum velocity = {min(velocity_results)/1000:.2f} km/s')
print(f'Finishing velocity = {velocity_results[-1]/1000:.2f} km/s')
print(f'Maximum deceleration = {max(deceleration_results):.2f} Gs')
index_of_max_decel = deceleration_results.index(max(deceleration_results))
print(f'Time of maximum deceleration = {time_results[index_of_max_decel]:.2f} s')


# Key stagnation point heating parameters
print('\n   Key Stagnation point heating parameters:')
print(f'Maximum S.P. convective heat flux = {max(q_conv_results):.2f} W/cm^2')
index_of_max_q_conv = q_conv_results.index(max(q_conv_results))
print(f'Time of maximum convective heat flux = {time_results[index_of_max_q_conv]:.2f} s')
print(f'Maximum S.P. radiative heat flux = {max(q_rad_results):.4f} W/cm^2')
index_of_max_q_rad = q_rad_results.index(max(q_rad_results))
print(f'Time of maximum radiative heat flux = {time_results[index_of_max_q_rad]:.2f} s')
print(f'Maximum S.P. total heat flux = {max(q_stag_total_results):.2f} W/cm^2')
index_of_max_q_stag_total = q_stag_total_results.index(max(q_stag_total_results))
print(f'Time of maximum S.P. heat flux = {time_results[index_of_max_q_stag_total]:.2f} s')
print(f'Maximum S.P. wall temperature = {max(Tw_results):.2f} K')
index_of_max_Tw = Tw_results.index(max(Tw_results))
print(f'Time of maximum S.P. wall temperature = {time_results[index_of_max_Tw]:.2f} s')

# Key local heating parameters
print('\n   Key local heating parameters:')
print(f'Maximum local total heat flux at {time_results[index_of_desired_time]:.2f} s= {max(local_heat_flux_list[index_of_desired_time]):.2f} W/cm^2')
print(f'Maximum local wall temperature at {time_results[index_of_desired_time]:.2f}' 
      's = {max(local_Tw_list[index_of_desired_time]):.2f} K')
print(f'Maximum total heat load = {max(heat_load_results):.2f} J/cm^2')