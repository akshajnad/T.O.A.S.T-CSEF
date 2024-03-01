import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.constants import k, physical_constants

def frc_boundary_conditions(ya, yb):
    center_bc = ya[1]
    edge_bc = yb[0]

    return np.array([center_bc, edge_bc])

def J_psi(psi, alpha, beta):
    # Example of a more complex J(psi) function
    # alpha: controls the linear response related to plasma pressure gradients
    # beta: introduces nonlinear effects, possibly representing kinetic phenomena or external field contributions
    return alpha * psi - beta * psi**2 + np.sin(psi) + 0.1 * np.cos(psi)

def grad_shafranov_equation(r, psi, alpha, beta):
    # Enhanced Grad-Shafranov equation with complex physics representations
    
    # Assuming psi[0] is the function and psi[1] is its derivative
    pressure_gradient = alpha * psi[0] - beta * psi[0]**2 + np.sin(psi[0]) + 0.1 * np.cos(psi[0])

    # Additional terms representing physics phenomena
    B_p = alpha * np.exp(-r**2) + beta * np.cos(r)
    B_t = psi[0]**2 / (1 + r**2)
    shafranov_shift = alpha / (1 + r**2 + psi[0]**2)
    J_phi = beta * np.sin(psi[0]) * np.exp(-psi[0]**2) + alpha * np.cos(psi[0])

    dpsi_dr = psi[1]
    d2psi_dr2 = -1/r * psi[1] + B_p * B_t + pressure_gradient + shafranov_shift + J_phi

    return np.vstack((dpsi_dr, d2psi_dr2))

def solve_for_different_parameters(alpha, beta):
    # Radial domain setup
    r = np.linspace(0.01, 1.0, 100)  # Avoiding r=0 due to the singularity
    psi_initial = np.zeros((2, r.size))

    # Wrap the Grad-Shafranov equation to include alpha and beta without changing its signature
    def equation_wrapped(r, psi):
        return grad_shafranov_equation(r, psi, alpha, beta)

    # Solve the BVP
    solution = solve_bvp(equation_wrapped, boundary_conditions, r, psi_initial)
    return solution.x, solution.y[0]

# Plotting for different values of alpha and beta
plt.figure(figsize=(10, 6))
params = [(1, .1), (5, .5), (10,1)]  # Example pairs of (alpha, beta)
for alpha, beta in params:
    r, psi = solve_for_different_parameters(alpha, beta)
    plt.plot(r, psi, label=f'Alpha = {alpha}, Beta = {beta}')

plt.xlabel('Radius $r$')
plt.ylabel('Magnetic Flux $\psi$')
plt.title('Magnetic Flux Profile with Complex J(ψ)')
plt.legend()
plt.grid(True)
plt.show()

def plasma_temperature(r, psi, alpha, beta):
    T_base = 1e6  # Base temperature in Kelvin
    psi_max_safe = np.max(psi) if np.max(psi) > 0 else 1e-5
    # Enhanced temperature model incorporating more complex physics
    T = T_base + (alpha * np.abs(psi)**2 + beta * np.sqrt(np.abs(psi))) / psi_max_safe
    return T


plt.figure(figsize=(10, 5))
for alpha, beta in params:
    r, psi = solve_for_different_parameters(alpha, beta)
    T = plasma_temperature(r, psi, alpha, beta)
    plt.plot(r, T, label='Plasma Temperature')

plt.xlabel('Radius $r$')
plt.ylabel('Temperature $T$ (Kelvin)')
plt.title('Plasma Temperature Distribution')
plt.legend()
plt.grid(True)
plt.show()


# Constants
k = 1.38e-23  # Boltzmann constant in J/K
Q = 1e-13  # Fusion energy release per reaction in J, assuming D-T fusion as an example
t = 1  # Energy confinement time in seconds

# Function to calculate the reaction rate <v*sigma> for D-T fusion, which has a peak around 14 keV

# Constants for D-T fusion reaction
deuteron_mass = 2.014102 * physical_constants['atomic mass constant'][0]  # Deuterium atomic mass in kg
triton_mass = 3.016049 * physical_constants['atomic mass constant'][0]  # Tritium atomic mass in kg
reduced_mass = (deuteron_mass * triton_mass) / (deuteron_mass + triton_mass)  # Reduced mass for the D-T system

def reaction_rate(T, T_peak):
    # Convert temperature from Kelvin to keV
    T_keV = T / 1.16045e7
    T_peak_keV = T_peak / 1.16045e7

    # Calculate the average velocity of the particles
    v = np.sqrt(2 * k * T / reduced_mass)

    # Placeholder for fusion cross-section as a function of temperature in keV
    sigma = 1.1e-24  # Placeholder value for sigma in m^2
    # Adjust sigma based on temperature, peaking at T_peak_keV
    sigma *= np.where(T_keV < T_peak_keV, (T_keV / T_peak_keV)**2, np.exp(T_peak_keV - T_keV))

    # The reaction rate is the product of v and sigma
    v_sigma = v * sigma

    return v_sigma

# Function to calculate plasma density based on the Lawson criterion
def plasma_density(T, T_peak):
    v_sigma = reaction_rate(T, T_peak)
    return 12 * k * T / (v_sigma * Q * t)

# Calculate T_peak across all alpha, beta pairs
T_peak = max([np.max(plasma_temperature(r, solve_for_different_parameters(alpha, beta)[1], alpha, beta)) for alpha, beta in params])

# Plotting
plt.figure(figsize=(10, 5))

for alpha, beta in params:
    r, psi = solve_for_different_parameters(alpha, beta)
    T = plasma_temperature(r, psi, alpha, beta)
    n = plasma_density(T, T_peak)
    plt.plot(r, n, label=f'Alpha = {alpha}, Beta = {beta}')

plt.xlabel('Radius $r$')
plt.ylabel('Plasma Density $n$ (particles/m^3)')
plt.title('Plasma Density Distribution Based on Lawson Criterion')
plt.legend()
plt.grid(True)
plt.show()
