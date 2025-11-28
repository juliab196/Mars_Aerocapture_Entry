# Mars Aerocapture Entry


This repository aims to assess the viability of aerocapture by implementing a three-degree-of-freedom trajectory 
simulation to model a non-thrusting, lifting entry with Mars aerocapture. 

The optimal trajectory is determined by varying the entry velocity, entry flight angle and lift-to-drag ratio within certain
ranges, in order to maximise the change in velocity (delta_v) while staying within deceleration, heating, altitude and flight 
angle constraints. 

Vehicle parameters including diameter, entry mass, drag coefficient, lift-to-drag-ratio, emissivity and heating limits were based on the Mars Science
Laboratory entry capsule which delivered NASA's Curiosity rover to the surface of Mars in 2012. The deceleration limit was taken 
from the allowable deceleration limit for a manned vehicle of 5 Gs (https://arc.aiaa.org/doi/abs/10.2514/3.26458?journalCode=jsr)

## Installation

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Run the optimization from the project root:
```bash
python main.py
```

This will:
1. Set up initial guesses for entry velocity, flight angle, and lift-to-drag ratio
2. Perform trajectory optimization to maximize delta-v subject to constraints
3. Generate diagnostic information about the optimal trajectory
4. Plot the results (altitude, velocity, deceleration, heating, etc.)

## Project Structure

```
Mars_Aerocapture_Entry/
├── main.py                          # Entry point; runs optimization and plotting
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── trajectory_optimizer/
    ├── __pycache__/                 # Python cache
    ├── constants.py                 # Physical parameters, constraint limits, and times
    ├── optimizer.py                 # Trajectory solver, post-processing, objective and constraints
    └── plot.py                      # Results visualization
```

## Key Parameters & Constraints

### Entry Conditions
- **Entry Velocity (v_e)**: 5000–7000 m/s
- **Entry Flight Angle (γ_e)**: −20° to −10°
- **Lift-to-Drag Ratio (L/D)**: 0.267–0.5

### Trajectory Constraints
- **Flight angle throughout trajectory**: −20° to +20°
- **Altitude range**: 20 km (minimum) to 200 km (maximum)
- **Maximum deceleration**: 5.0 G
- **Maximum heat flux**: 450 kW/m²
- **Maximum wall temperature**: 1500 K
- **Maximum total heat load**: 3500 kJ/m²

### Optimization Objective
- Maximize change in velocity (Δv = v_entry − v_final) during aerocapture

## Model Details

### Equations of Motion
The 3-DOF trajectory model integrates the following differential equations:
- Velocity rate: governed by drag and lift components
- Flight angle rate: governed by lift, drag, and gravity
- Range and altitude: integrated from velocity components

### Numerical Methods
- **Integrator**: SciPy's `solve_ivp` with RK45 method
- **Optimizer**: SciPy's `minimize` with SLSQP algorithm
- **Integration tolerance**: rtol=1e-6, atol=1e-6
- **Maximum step size**: 0.05 s

### Post-Processing
For each trajectory, convective and radiative heating rates are computed, and wall temperature and deceleration are estimated based on energy balance principles.

## References

- Mars Science Laboratory Entry, Descent, and Landing: 
    Karl T. Edquist, A. A. (2009). Aerothermodynamic Design of the Mars Science Laboratory Heatshield. American Institute of Aeronautics and Astronautics(41).
    Karl T. Edquist, P. N. (2011). Aerodynamics for Mars Phoenix Entry Capsule. Journal of Spacecraft and Rockets, 48, 713-726 . Retrieved 2024
- Deceleration Limits for Human Spaceflight: https://arc.aiaa.org/doi/abs/10.2514/3.26458?journalCode=jsr
- Heating relations: Tauber, M. E. (1991). Stagnation-point radiative heating relations for Earth and Mars entries.
Journal of Spacecraft and Rockets, 28, 40–42.
- Mars fact sheet: Williams, D. D. (2024). Mars Fact Sheet. Retrieved from NASA Official:
https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
- Atmospheric Models: Mars-GRAM 2010

## License

This project is part of AERO4800 (Aerospace Engineering capstone course).
