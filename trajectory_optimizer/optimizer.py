from math import sin, cos, exp, pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from trajectory_optimizer.constants import *

# Global iteration counter
iteration_count = 0

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
    return h - 120000.0 

# Do not terminate integration when crossing 120 km so trajectories run to `final_time`.
# We still capture the event (solution.t_events) if desired, but allow full integration.
exit_atmosphere.terminal = False
exit_atmosphere.direction = -1.0

# Stop if altitude goes unexpectedly high (safety event)
def altitude_too_high(t, y):
    h = y[3]
    return h - 200000.0

altitude_too_high.terminal = True
altitude_too_high.direction = 1.0


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
                    events=[exit_atmosphere, altitude_too_high])
    # Mark on the solution if the altitude_too_high event fired (events index 1)
    try:
        solution.altitude_too_high = len(solution.t_events[1]) > 0
    except Exception:
        solution.altitude_too_high = False

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

    # Calculate convective heat flux (West & Brandis CFD Curve fit empirical correlation)
    q_conv = 7.2074 * (rho ** 0.4739) * ((Diam / 2) ** -0.5405) * ((velocity[:-1] / 1000) ** 3.4956)

    # Calculate radiative heat flux (Tauber & Sutton CFD Curve fit empirical correlation)
    func_v = np.where(velocity[:-1] >= 6000, 0.2, 0.0)
    C = 2.35e4
    a, b = 0.525, 1.19
    q_rad = C * ((Diam / 2) ** a) * (rho ** b) * func_v

    # Calculate total stagnation point heat flux
    q_stag_total = q_conv + q_rad
    # Calculate stagnation point wall temperature, assuming no conduction occurs into the material (therefore 
    # the stagnation point heat flux in is matched by the re-radiation), therefore finding the theoretical max 
    # wall temperature
    Tw = ((q_stag_total) * 10000 / (sb * e_w)) ** 0.25

    # Calculate deceleration (Gs)
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
    # Broadcast these to (N-1, M-1)

    # Reshape arrays for broadcasting
    rho_col = rho[:, np.newaxis]         # shape (N-1, 1)
    velocity_col = velocity[:-1][:, np.newaxis]  # shape (N-1,1)
    alpha_row = alpha_list[1:][np.newaxis, :]    # shape (1, M-1)
    x_row = x_list[1:][np.newaxis, :]            # shape (1, M-1)

    # Local_heat_flux array shape (N-1, M-1)
    local_heat_flux = (9.43e-5) * np.sqrt((rho_col * np.cos(alpha_row) * np.sin(alpha_row)) / x_row) * velocity_col ** 3  # W/m^2    
    local_heat_flux /= 10000 # Convert W/m^2 to W/cm^2

    # Local wall temperature
    local_Tw = (local_heat_flux / (sb * e_w)) ** 0.25  # shape (N-1, M-1)

    local_heat_flux_list = local_heat_flux.tolist()
    local_Tw_list = local_Tw.tolist()

    # Total heat load across entire trajectory for each x segment (integrate over time intervals)
    heat_load_results = np.zeros(x_segments)

    # Only sum over segments 1 to end (skip stagnation point x=0 to align with local flux shape)
    heat_load_results[1:] = np.sum(local_heat_flux * dt_array[:, np.newaxis], axis=0)

    global iteration_count
    iteration_count += 1
    print(f"Iteration {iteration_count}")

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
# GOALS: maximize delta-v while staying within heating, decceleration, flight angle and altitude limits
def objective_delta_v(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    # If safety event triggered (altitude too high), penalize heavily so optimizer rejects this
    if getattr(solution, 'altitude_too_high', False):
        return 1e6
    velocity = solution.y[0]
    v_final = velocity[-1]
    delta_v = v_e - v_final 
    return -delta_v

def heat_flux_constraint(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    if getattr(solution, 'altitude_too_high', False):
        return -1e6
    q_stag_total_results = perform_post_processing(solution, cL)["stagnation_heat_flux"]
    max_heat_flux_actual = max(q_stag_total_results)
    return max_heat_flux - max_heat_flux_actual # must be > 0 

def wall_temp_constraint(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    if getattr(solution, 'altitude_too_high', False):
        return -1e6
    Tw_results = perform_post_processing(solution, cL)["wall_temperatures"]
    max_Tw_actual = max(Tw_results)
    return max_Tw - max_Tw_actual # must be > 0 

def total_heat_load_constraint(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    if getattr(solution, 'altitude_too_high', False):
        return -1e6
    heat_load_results = perform_post_processing(solution, cL)["total_heat_load"]
    max_total_heat_load_actual = max(heat_load_results)
    return max_total_heat_load - max_total_heat_load_actual # must be > 0

def deceleration_constraint(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e] 
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    if getattr(solution, 'altitude_too_high', False):
        return -1e6
    decel_results = perform_post_processing(solution, cL)["deceleration"]
    max_decel_actual = max(decel_results)
    return max_decel - max_decel_actual # must be > 0

def minimum_altitude_constraint(opt_variables):
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e]
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    if getattr(solution, 'altitude_too_high', False):
        return -1e6
    altitude_results = perform_post_processing(solution, cL)["altitude"]
    min_altitude_actual = min(altitude_results)
    return min_altitude_actual - 20000  # must be > 0 (minimum 20 km)

def flight_angle_constraint(opt_variables):
        """
        Enforce two things:
        - Entry flight path angle (gamma_e) must be within [-20, -10] degrees (this is enforced
            via the optimizer bounds, but keep as a constraint for robustness).
        - The flight path angle throughout the trajectory must remain within [-20, +20] degrees.
        Note: optimizer variables use radians for angles; convert as needed.
        """
        v_e, gamma_e, LD = opt_variables

        # Check entry angle (gamma_e) in degrees
        gamma_entry_deg = np.degrees(gamma_e)
        margin_entry_lower = gamma_entry_deg + 20.0    # >=0 when gamma_entry_deg >= -20
        margin_entry_upper = -10.0 - gamma_entry_deg    # >=0 when gamma_entry_deg <= -10

        # Now check trajectory gamma stays within [-20, +20] deg
        initial_conditions = [v_e, gamma_e, x_e, h_e]
        solution, cL = solve_trajectory_IVP(initial_conditions, LD)
        if getattr(solution, 'altitude_too_high', False):
            return -1e6
        gamma_traj = solution.y[1]  # radians
        gamma_deg_traj = np.degrees(gamma_traj)
        max_gamma = np.max(gamma_deg_traj)
        min_gamma = np.min(gamma_deg_traj)
        margin_traj_upper = 20.0 - max_gamma  # >=0 when max_gamma <= 20
        margin_traj_lower = min_gamma + 20.0  # >=0 when min_gamma >= -20

        # All margins must be positive. Return the smallest margin (inequality constraint expects > 0)
        return min(margin_entry_lower, margin_entry_upper, margin_traj_upper, margin_traj_lower)

def trajectory_validity_constraint(opt_variables):
    # Ensure trajectory stays within altitude bounds
    v_e, gamma_e, LD = opt_variables
    initial_conditions = [v_e, gamma_e, x_e, h_e]
    solution, cL = solve_trajectory_IVP(initial_conditions, LD)
    altitude_trajectory = solution.y[3]
    max_altitude = np.max(altitude_trajectory)
    # Should not exceed 200km
    return 200000.0 - max_altitude  # must be > 0 

def optimize(opt_variables):
    # Bounds for optimizer variables: velocity (m/s), entry flight path angle (radians), L/D ratio
    # Note: optimizer uses radians for angles, so convert degree bounds to radians here.
    bounds = [
        (3000, 8000),  # v_e
        (np.radians(-20.0), np.radians(-10.0)),  # gamma_e bounds in radians
        (0.267, 0.5)  # L/D
    ]
    constraints = [
        {'type': 'ineq', 'fun': heat_flux_constraint},
        {'type': 'ineq', 'fun': wall_temp_constraint},
        {'type': 'ineq', 'fun': total_heat_load_constraint},
        {'type': 'ineq', 'fun': deceleration_constraint},
        {'type': 'ineq', 'fun': minimum_altitude_constraint},
        {'type': 'ineq', 'fun': flight_angle_constraint},
        {'type': 'ineq', 'fun': trajectory_validity_constraint}]
    result = minimize(objective_delta_v, opt_variables, bounds=bounds, constraints=constraints, options={'disp': True})

    return result
