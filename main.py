from trajectory_optimizer.optimizer import optimize, solve_trajectory_IVP, perform_post_processing
from trajectory_optimizer.constants import *
from trajectory_optimizer.plot import plot
import numpy as np

"""INITAL OPTIMAL GUESS"""
v_e = 6000                       # entry velocity (m/s)
gamma_e = np.radians(-14)        # entry flight angle (deg)
LD = 0.4                         # lift-to-drag-ratio 
opt_variables = [v_e, gamma_e, LD]

"""PERFORM OPTIMIZATION"""
result = optimize(opt_variables)
# Extract solution
v_opt, gamma_opt_rad, LD_opt = result.x
gamma_opt_deg = np.degrees(gamma_opt_rad)
delta_v_opt = -result.fun

"""DISPLAY OPTIMIZATION RESULTS"""
print(f"\nOptimization Results:")
print(f"Optimal entry velocity (m/s): {v_opt:.2f}")
print(f"Optimal flight path entry angle (deg): {gamma_opt_deg:.2f}")
print(f"Optimal L/D ratio: {LD_opt:.2f}")
print(f"Maximum achievable delta-v (m/s): {delta_v_opt:.2f}")
# --- Diagnostic run of the optimal trajectory ---
initial_conditions_opt = [v_opt, gamma_opt_rad, x_e, h_e]
solution_opt, cL_opt = solve_trajectory_IVP(initial_conditions_opt, LD_opt)
print('\nOptimal trajectory diagnostics:')
print(f"  Integration finished at t = {solution_opt.t[-1]:.2f} s (requested final_time = {final_time} s)")
print(f"  Solver success flag: {solution_opt.success}, status: {solution_opt.status}")
print(f"  Number of time points: {len(solution_opt.t)}")
print(f"  Altitude-too-high event triggered: {getattr(solution_opt, 'altitude_too_high', False)}")
try:
	print(f"  Event times: {[ev.tolist() for ev in solution_opt.t_events]}")
except Exception:
	pass

pp = perform_post_processing(solution_opt, cL_opt)
alt = np.array(pp['altitude'])
vel = np.array(pp['velocity'])
gamma_traj = np.degrees(np.array(solution_opt.y[1]))
decel = np.array(pp['deceleration'])
print(f"  Min altitude = {np.min(alt)/1000:.2f} km at t ~ {solution_opt.t[np.argmin(alt)]:.2f} s")
print(f"  Max altitude = {np.max(alt)/1000:.2f} km at t ~ {solution_opt.t[np.argmax(alt)]:.2f} s")
print(f"  Min velocity = {np.min(vel)/1000:.3f} km/s")
print(f"  Max velocity = {np.max(vel)/1000:.3f} km/s")
print(f"  Min flight angle = {np.min(gamma_traj):.2f} deg, Max flight angle = {np.max(gamma_traj):.2f} deg")
print(f"  Max deceleration = {np.max(decel):.2f} Gs at t ~ {solution_opt.t[np.argmax(decel)]:.2f} s")

# Pass the computed solution and post-processing to the plot function
plot(LD_opt, v_opt, gamma_opt_rad, x_e, h_e, solution=solution_opt, post_processing_results=pp)
