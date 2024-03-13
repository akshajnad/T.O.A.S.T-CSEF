import numpy as np
from scipy.optimize import minimize

def objective_function(params):
    alpha, beta, radius_end = params
    if radius_end <= 0.01:
        return np.inf  # Return a large number to penalize invalid solutions

    r = np.linspace(0.01, radius_end, 100)
    if np.any(np.diff(r) <= 0):
        return np.inf  # Ensuring 'r' is strictly increasing, else penalize

    _, psi = solve_for_different_parameters(alpha, beta, radius_end)
    T = plasma_temperature(r, psi, alpha, beta)
    n = plasma_density(T, np.max(T))

    # Calculate penalties and objective value
    temp_penalty = np.sum(np.where(T > T_max, (T - T_max)**2, 0))
    density_penalty = np.sum(np.where(n > n_max, (n - n_max)**2, 0))
    alpha_penalty = 0 if 5 <= alpha <= 10 else (min(abs(alpha - 5), abs(alpha - 10)))**15
    beta_penalty = 0 if 0.01 <= beta <= 0.2 else (min(abs(beta - 0.01), abs(beta - 0.2)))**7

    return -np.mean(T) - np.mean(n) + temp_penalty + density_penalty + 100 * alpha_penalty + beta_penalty

bounds = [(.1, 20), (.01, .5), (0.05, 2.0)]  # Adjusted bounds for radius_end
optimized_params_all = []

# Initial guess and optimization
for y in range(100):
    initial_guess = [np.random.uniform(.1, 20), np.random.uniform(0.01, .5), np.random.uniform(0.05, 2.0)]
    result = minimize(objective_function, initial_guess, bounds=bounds, method='TNC')
    optimized_params_all.append(result.x)

optimized_params_all = np.array(optimized_params_all)

# Calculate the average of the optimized parameters across all iterations
average_optimized_params = np.mean(optimized_params_all, axis=0)

print("Average optimized parameters:", average_optimized_params)



def optimize_current_pulse(x, alpha, beta, r_end, T_max, density_max):
    current, pulse_duration = x
    T_mean, density_mean = model_current_and_pulse(current, pulse_duration, alpha, beta, r_end)
    # Penalties for exceeding maximum temperature and density
    temp_penalty = 0 if T_mean <= T_max else (T_mean - T_max)**2
    density_penalty = 0 if density_mean <= density_max else (density_mean - density_max)**2
    # Realistic penalties for current and pulse duration
    current_penalty = 0 if 400 <= current <= 3000 else (3000 - current)**15 if current > 3000 else (400 - current)**15
    pulse_penalty = 0 if 5 <= pulse_duration <= 50 else (50 - pulse_duration)**7 if pulse_duration > 50 else (5 - pulse_duration)**7

    # Objective function combines all factors
    return temp_penalty + density_penalty + 100 * current_penalty + 1000 * pulse_penalty
    
# Temperature and density maximum thresholds
T_max = 150e6  # 150 million Kelvin
density_max = 1e22  # Max density in particles/m^3
alpha_example = 102.38770799
beta_example = 22.95855441
r_end_example = 13.5762052

# Optimization bounds for current and pulse duration
bounds = [(100, 5000), (.5, 200)]  # Example bounds, adjust as needed

# Initial guess for the optimization

# Perform the optimization
optimized_params_magnets = []
for x in range(100):
    initial_guess = [np.random.uniform(100, 5000), np.random.uniform(.5, 200)]
    result = minimize(optimize_current_pulse, initial_guess, args=(alpha_example, beta_example, r_end_example, T_max, density_max), bounds=bounds, method='TNC')
    optimized_params_magnets.append(result.x)

optimized_params_magnets = np.array(optimized_params_magnets)
average_optimized_params_magnets = np.mean(optimized_params_magnets, axis=0)

print("Average optimized parameters:", average_optimized_params)
