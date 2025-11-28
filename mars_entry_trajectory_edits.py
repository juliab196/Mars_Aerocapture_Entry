#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:37:47 2024

@author: juliabaird
"""

import matplotlib.pyplot as plt
from math import sin, cos, exp, pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# variables to change: entry velocity, flight path entry angle, lift over drag ratio
# fixed variables: ballistic coefficient, entry mass
# Optimization objective: maximise delta v while staying within heating and deceleration load limits and 
    # Deceleration limit: 5g
    # Heating load limits: Max Heat flux = 225 W/cm^2, Max total heat load = 5477 J/cm^2, 
    #                      Max Twall = 1873 K 


#------------------------------------------------------------------------------
"""Define Parameters"""

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
initial_time = 0.0 # s
final_time = 500.0 # s

"""Define Constraint Limits"""
max_heat_flux = 225              # maximum allowable heat flux (W/cm^2)
max_Tw = 1873                    # maximum allowable TPS wall temp (K)
max_total_heat_load = 5477       # maximum allowable total heat laod (J/cm^2)
max_decel = 5      # Gs

#------------------------------------------------------------------------------
"""Flight Trajectory Differential Equations"""

def f(t, y, cL):
    
    # define v, gamma, x and h
    v, gamma, x, h = y
    
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

# Include a stopping criteria when Vehicle exits atmosphere
def exit_atmosphere(t, y):
    h = y[3]
    return 120000.0 - h 

exit_atmosphere.terminal = True
exit_atmosphere.direction = 1.0

#------------------------------------------------------------------------------
"""Perform Trajectory Numerical Integration"""

def solve_trajectory_IVP(initial_conditions, LD):

    steps_to_evaluate = 10000
    rtol = 1.0e-6
    atol = 1.0e-6
    cL = LD * cD
    solution = solve_ivp(lambda t, y: f(t, y, cL),
                    t_span = [initial_time, final_time], 
                    y0 = initial_conditions, 
                    method='RK45', 
                    max_step = 0.05,
                    rtol = rtol, atol = atol,
                    dense_output = True, 
                    events = lambda t, y: exit_atmosphere(t, y))

    return solution, cL


def perform_post_processing(solution, cL):
    # Extract solution 
    time = solution.t
    dt_array = np.diff(time)  # length N-1
    y_results = solution.y
    velocity = y_results[0]
    altitude = y_results[3]

    # Calculate local density for each time step except last (per dt_array length)
    rho = rho_s * np.exp(-altitude[:-1] / beta_neg)  # shape (N-1,)

    # Convective heat flux
    q_conv = 7.2074 * (rho ** 0.4739) * ((Diam / 2) ** -0.5405) * ((velocity[:-1] / 1000) ** 3.4956)

    # Radiative heat flux: vectorize condition
    func_v = np.where(velocity[:-1] >= 6000, 0.2, 0.0)
    C = 2.35e4
    a, b = 0.525, 1.19
    q_rad = C * ((Diam / 2) ** a) * (rho ** b) * func_v

    # Total stagnation point heat flux and wall temperature
    q_stag_total = q_conv + q_rad
    Tw = ((q_stag_total) * 10000 / (sb * e_w)) ** 0.25

    # Deceleration (Gs)
    Lift = 0.5 * rho * (velocity[:-1] ** 2) * S * cL
    Drag = 0.5 * rho * (velocity[:-1] ** 2) * S * cD
    deceleration = (0.5 * rho * (velocity[:-1] ** 2) * np.sqrt(1 + (Lift / Drag) ** 2) * (S * cD / m_e)) / 9.81

    # Local surface locations and angles
    shell_angle = np.radians(70)
    shell_h = (Diam / 2) / np.tan(shell_angle)
    x_length = (Diam / 2) / np.sin(shell_angle)
    x_segments = 100
    x_list = np.linspace(0, x_length, x_segments)

    side_length = np.sqrt(shell_h ** 2 + x_list ** 2 - 2 * shell_h * x_list * np.cos(shell_angle))
    alpha_list = np.arcsin(x_list * np.sin(shell_angle) / side_length)

    # Compute local heat flux for all time steps and x segments (skip the first x=0 to avoid div by zero)
    # rho: shape (N-1,), alpha_list[1:], x_list[1:] shape (M-1,)
    # We want to broadcast these to (N-1, M-1)

    # Reshape arrays for broadcasting
    rho_col = rho[:, np.newaxis]         # shape (N-1, 1)
    velocity_col = velocity[:-1][:, np.newaxis]  # shape (N-1,1)
    alpha_row = alpha_list[1:][np.newaxis, :]    # shape (1, M-1)
    x_row = x_list[1:][np.newaxis, :]            # shape (1, M-1)

    # local_heat_flux array shape (N-1, M-1)
    local_heat_flux = (9.43e-5) * np.sqrt((rho_col * np.cos(alpha_row) * np.sin(alpha_row)) / x_row) * velocity_col ** 3  # W/m^2

    # Convert W/m^2 to W/cm^2 by dividing by 10^4
    local_heat_flux /= 10000

    # Local wall temperature
    local_Tw = (local_heat_flux / (sb * e_w)) ** 0.25  # shape (N-1, M-1)

    local_heat_flux_list = local_heat_flux.tolist()
    local_Tw_list = local_Tw.tolist()

    # Total heat load across entire trajectory for each x segment (integrate over time intervals)
    heat_load_results = np.zeros(x_segments)

    # Only sum over segments 1 to end (skip stagnation point x=0 to align with local flux shape)
    heat_load_results[1:] = np.sum(local_heat_flux * dt_array[:, np.newaxis], axis=0)

    print("Post processing complete.")

    results = {
        "time": time.tolist(),
        "velocity": velocity.tolist(),
        "altitude": altitude.tolist(),
        "wall_temperatures": Tw.tolist(),
        "stagnation_heat_flux": q_stag_total.tolist(),
        "convective_heat_flux": q_conv.tolist(),
        "radiative_heat_flux": q_rad.tolist(),
        "total_heat_load": heat_load_results.tolist(),
        "deceleration": deceleration.tolist(),
        "x_list": x_list.tolist(),
        "local_heat_flux": local_heat_flux_list,
        "local_wall_temp": local_Tw_list,
    }
    return results


#------------------------------------------------------------------------------
"""Optimize Trajectory Problem"""
# GOALS: maximize delta-v while staying within heating and acceleration limits
def objective_delta_v(opt_variables):

    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    velocity = solution.y[0]
    v_final = velocity[-1]
    delta_v = v_e - v_final 

    return -delta_v

def heat_flux_constraint(opt_variables):

    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    q_stag_total_results = perform_post_processing(solution, cL)["stagnation_heat_flux"]
    max_heat_flux_actual = max(q_stag_total_results)

    return max_heat_flux - max_heat_flux_actual # must be >0 

def wall_temp_constraint(opt_variables):

    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    Tw_results = perform_post_processing(solution, cL)["wall_temperatures"]
    max_Tw_actual = max(Tw_results)

    return max_Tw - max_Tw_actual # must be > 0 

def total_heat_load_constraint(opt_variables):

    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    heat_load_results = perform_post_processing(solution, cL)["total_heat_load"]
    max_total_heat_load_actual = max(heat_load_results)

    return max_total_heat_load - max_total_heat_load_actual # must be > 0

def deceleration_constraint(v_e, gamma_e, LD):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    decel_results = perform_post_processing(solution, cL)["deceleration"]
    max_decel_actual = max(decel_results)
    return max_decel - max_decel_actual # must be > 0 

# Bounds for optimizer variables: velocity (m/s), flight path angle (deg), L/D ratio
bounds = [(4000, 7000),  # v_e
          (-20, -10),     # gamma_deg
          (0.1, 0.6)]    # L/D

constraints = [
    {'type': 'ineq', 'fun': heat_flux_constraint},
    {'type': 'ineq', 'fun': wall_temp_constraint},
    {'type': 'ineq', 'fun': total_heat_load_constraint},
    {'type': 'ineq', 'fun': deceleration_constraint}]

# Inital guess
v_e = 6000               # entry velocity (m/s)
gamma_e = -11.5          # entry flight angle (deg)
LD = 0.4
opt_variables = [v_e, np.radians(gamma_e), LD]
    
# Perform optimization
result = minimize(objective_delta_v, opt_variables, bounds=bounds, constraints=constraints, options={'disp': True})

v_opt, gamma_opt_deg, LD_opt = result.x
delta_v_opt = -result.fun

print(f"\nOptimization Results:")
print(f"Optimal entry velocity (m/s): {v_opt:.2f}")
print(f"Optimal flight path angle (deg): {gamma_opt_deg:.2f}")
print(f"Optimal L/D ratio: {LD_opt:.2f}")
print(f"Maximum achievable delta-v (m/s): {delta_v_opt:.2f}")


#------------------------------------------------------------------------------
"""Post Processing"""
opt_initial_conditions = [v_opt, gamma_opt_deg, x_e, h_e]
opt_solution, opt_cL = solve_trajectory_IVP(opt_initial_conditions, LD_opt)

# Extract results from numerical integration
time_results = opt_solution.t
velocity_results = opt_solution.y[0]
raw_gamma_results = opt_solution.y[1]
gamma_results = np.degrees(raw_gamma_results)
down_range_results = opt_solution.y[2]
altitude_results = opt_solution.y[3]

post_processing_results = perform_post_processing(opt_solution, opt_cL)

Tw_results = post_processing_results["wall_temperatures"]
q_stag_total_results = post_processing_results["stagnation_heat_flux"]
heat_load_results = post_processing_results["total_heat_load"]
decel_results = post_processing_results["deceleration"]
x_list = post_processing_results["x_list"]
local_heat_flux_list = post_processing_results["local_heat_flux"]
local_Tw_list = post_processing_results["local_wall_temp"]
q_conv_results = post_processing_results["convective_heat_flux"]
q_rad_results = post_processing_results["radiative_heat_flux"]


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
ax.plot(time_results, decel_results)
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
print(f'Maximum deceleration = {max(decel_results):.2f} Gs')
index_of_max_decel = decel_results.index(max(decel_results))
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