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
for y in range(5):
    initial_guess = [np.random.uniform(.1, 20), np.random.uniform(0.01, .5), np.random.uniform(0.05, 2.0)]
    result = minimize(objective_function, initial_guess, bounds=bounds, method='TNC')
    optimized_params_all.append(result.x)

optimized_params_all = np.array(optimized_params_all)

# Calculate the average of the optimized parameters across all iterations
average_optimized_params = np.mean(optimized_params_all, axis=0)

print("Average optimized parameters:", average_optimized_params)
