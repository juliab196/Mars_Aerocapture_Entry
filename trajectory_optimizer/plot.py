import matplotlib.pyplot as plt
import os
import numpy as np
from trajectory_optimizer.optimizer import solve_trajectory_IVP, perform_post_processing

"""Plotting"""
def plot(LD_opt, v_opt, gamma_opt_rad_e,  x_e, h_e, solution=None, post_processing_results=None):

    # If caller provided a precomputed solution and post-processing results, use them
    if solution is None or post_processing_results is None:
        opt_initial_conditions = [v_opt, gamma_opt_rad_e, x_e, h_e]
        opt_solution, opt_cL = solve_trajectory_IVP(opt_initial_conditions, LD_opt)
        post_processing_results = perform_post_processing(opt_solution, opt_cL)
    else:
        opt_solution = solution
        opt_cL = None

    # Extract results from numerical integration
    time_results = opt_solution.t
    velocity_results = opt_solution.y[0]
    raw_gamma_results = opt_solution.y[1]
    gamma_results = np.degrees(raw_gamma_results)
    down_range_results = opt_solution.y[2]
    altitude_results = opt_solution.y[3]

    # post_processing_results is already set above when needed; do not recompute if provided

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
    FIG_DIR = "figures_mars_aerocapture"
    os.makedirs(FIG_DIR, exist_ok=True)

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
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    # Convective and radiative heating plot
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(time_results[:-1], q_conv_results,'tab:green')
    ax[0].set_title('Time vs S.P. Convective Heat Flux')
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_ylabel('Convective Heat Flux (W/cm^2)')
    ax[1].plot(time_results[:-1], q_rad_results, 'tab:orange')
    ax[1].set_title('Time vs S.P. Radiative Heat Flux')
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_ylabel('Radiative Heat Flux (W/cm^2)')
    plt.tight_layout()
    figure_filename = 'Convective and radiative heating'
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    # Velocity vs altitude plot
    fig, ax = plt.subplots()
    ax.plot(velocity_results/1000, altitude_results/1000.0, 'tab:blue')
    ax.set_title('Velocity vs Altitude')
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Altitude (km)')
    plt.tight_layout()
    figure_filename = 'Velocity vs Altitude'
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    # Deceleration plot
    fig, ax = plt.subplots()
    ax.plot(time_results[:-1], decel_results)
    ax.set_title('Time vs Deceleration')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel("G's")
    plt.tight_layout()
    figure_filename = 'Deceleration'
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    # Stagnation point heat flux and wall temperature plot
    fig, ax = plt.subplots(1,2)

    ax[0].plot(time_results[:-1], q_stag_total_results, 'tab:purple')
    ax[0].set_title('Time vs S.P. Heat Flux')
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_ylabel('Heat Flux (W/cm^2)')
    ax[1].plot(time_results[:-1], Tw_results, 'tab:red')
    ax[1].set_title('Time vs S.P. Wall Temperature ')
    ax[1].set_xlabel('Time (sec)')
    ax[1].set_ylabel('Wall Temperature (K)')
    plt.tight_layout()
    figure_filename = 'Stagnation point heat flux and temperature'
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    # Local Heat flux and wall temperature plot
    # Guard index_of_desired_time to be within available time steps for local arrays (length N-1)
    max_time_index = min(len(time_results)-2, len(local_heat_flux_list)-1)
    index_of_desired_time = max(0, max_time_index // 2)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(x_list[1:], local_heat_flux_list[index_of_desired_time], 'tab:purple')
    ax[0].set_title(f'Local Heat Flux at {time_results[index_of_desired_time]:.1f}s')
    ax[0].set_xlabel('Distance from stagnation point (m)')
    ax[0].set_ylabel('Heat Flux (W/cm^2)')
    ax[1].plot(x_list[1:], local_Tw_list[index_of_desired_time], 'tab:red')
    ax[1].set_title(f'Local Wall Temp. at {time_results[index_of_desired_time]:.1f}s')
    ax[1].set_xlabel('Distance from stagnation point (m)')
    ax[1].set_ylabel('Temperature (K)')
    plt.tight_layout()
    figure_filename = 'Local heat flux and local wall temperature '
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
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
    plt.savefig(os.path.join(FIG_DIR, figure_filename + '.png'))
    plt.show()

    #------------------------------------------------------------------------------
    """Extract Key Parameters"""

    # Key trajectory parameters
    print('\n   Key trajectory parameters:')
    print(f'Minimum altitude = {min(altitude_results)/1000:.2f} km')
    index_of_min_alt = np.argmin(altitude_results)
    print(f'Time of minimum altitude = {time_results[index_of_min_alt]:.2f} s')
    print(f'Flight angle at minimum altitude = {gamma_results[index_of_min_alt]:.2f} deg')
    print(f'Velocity at at minimum altitude = {velocity_results[index_of_min_alt]/1000:.2f} km/s')
    print(f'Finishing altitude = {altitude_results[-1]/1000:.2f} km')
    print(f'Maximum flight angle = {max(gamma_results):.2f} deg')
    index_of_max_gamma = np.argmax(gamma_results)
    print(f'Time of maximum flight angle = {time_results[index_of_max_gamma]:.2f} s')
    print(f'Finishing flight angle = {gamma_results[-1]:.2f} deg')
    print(f'Maximum down range distance = {max(down_range_results)/1000:.2f} km')
    print(f'Minimum velocity = {min(velocity_results)/1000:.2f} km/s')
    print(f'Finishing velocity = {velocity_results[-1]/1000:.2f} km/s')
    print(f'Maximum deceleration = {max(decel_results):.2f} Gs')
    index_of_max_decel = np.argmax(decel_results)
    # decel_results corresponds to time_results[:-1]
    print(f'Time of maximum deceleration = {time_results[:-1][index_of_max_decel]:.2f} s')


    # Key stagnation point heating parameters
    print('\n   Key Stagnation point heating parameters:')
    print(f'Maximum S.P. convective heat flux = {max(q_conv_results):.2f} W/cm^2')
    index_of_max_q_conv = np.argmax(q_conv_results)
    print(f'Time of maximum convective heat flux = {time_results[:-1][index_of_max_q_conv]:.2f} s')
    print(f'Maximum S.P. radiative heat flux = {max(q_rad_results):.4f} W/cm^2')
    index_of_max_q_rad = np.argmax(q_rad_results)
    print(f'Time of maximum radiative heat flux = {time_results[:-1][index_of_max_q_rad]:.2f} s')
    print(f'Maximum S.P. total heat flux = {max(q_stag_total_results):.2f} W/cm^2')
    index_of_max_q_stag_total = np.argmax(q_stag_total_results)
    print(f'Time of maximum S.P. heat flux = {time_results[:-1][index_of_max_q_stag_total]:.2f} s')
    print(f'Maximum S.P. wall temperature = {max(Tw_results):.2f} K')
    index_of_max_Tw = np.argmax(Tw_results)
    print(f'Time of maximum S.P. wall temperature = {time_results[:-1][index_of_max_Tw]:.2f} s')

    # Key local heating parameters
    print('\n   Key local heating parameters:')
    print(f'Maximum local total heat flux at {time_results[:-1][index_of_desired_time]:.2f} s = {max(local_heat_flux_list[index_of_desired_time]):.2f} W/cm^2')
    print(f'Maximum local wall temperature at {time_results[:-1][index_of_desired_time]:.2f} s = {max(local_Tw_list[index_of_desired_time]):.2f} K')
    print(f'Maximum total heat load = {max(heat_load_results):.2f} J/cm^2')

