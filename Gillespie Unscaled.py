import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Adjust parameters for your own model.
r = 1.0
K = 100
a = 0.01
s = 1.0
k = 0.1
N0 = 50
d = 0.2
Omega = 100

# Equilibrium values, if you have reference values and want to plot
N_star = N0 + np.arctanh(d / s) / k
P_star = (r / a) * (1 - N_star / K)

# Runs a single Gillespie
def gillespie_scaled(N0_val, P0_val, T_max, dt_sample=0.1):
    t = 0
    N = int(N0_val * Omega)
    P = int(P0_val * Omega)

    times = [0.0]
    N_series = [N / Omega]
    P_series = [P / Omega]
    next_sample_time = dt_sample

    while t < T_max and N > 0 and P > 0:
        # Rename variables for your model
        fish_birth = r * N * (1 - N / (K * Omega))
        fish_death = a * N * P / Omega
        fisher_birth = max(P * s * np.tanh(k * (N / Omega - N0)), 0)
        fisher_death = d * P

        a0 = fish_birth + fish_death + fisher_birth + fisher_death
        if a0 <= 0:
            break

        # Uses the direct method
        dt = np.random.exponential(1 / a0)
        t += dt

        r2 = np.random.uniform(0, a0)
        if r2 < fish_birth:
            N += 1
        elif r2 < fish_birth + fish_death:
            N = max(N - 1, 0)
        elif r2 < fish_birth + fish_death + fisher_birth:
            P += 1
        else:
            P = max(P - 1, 0)

        while next_sample_time <= t and next_sample_time <= T_max:
            times.append(next_sample_time)
            N_series.append(N / Omega)
            P_series.append(P / Omega)
            next_sample_time += dt_sample

    return np.array(N_series), np.array(P_series)

# Adjust settings for your own model
num_runs = 100
T_max = 100
dt_sample = 1.0
time_grid = np.arange(0, T_max + dt_sample, dt_sample)
num_steps = len(time_grid)

# All in One Function for JobLib
def run_one_sim():
    N_vals, P_vals = gillespie_scaled(0.6, 0.2, T_max, dt_sample)
    if len(N_vals) < num_steps:
        N_vals = np.pad(N_vals, (0, num_steps - len(N_vals)), constant_values=N_vals[-1])
        P_vals = np.pad(P_vals, (0, num_steps - len(P_vals)), constant_values=P_vals[-1])
    return N_vals, P_vals

# Parallel Computing
results = Parallel(n_jobs=10, backend="loky")(delayed(run_one_sim)() for _ in range(num_runs))
N_matrix, P_matrix = zip(*results)
N_avg = np.mean(N_matrix, axis=0)
P_avg = np.mean(P_matrix, axis=0)

# Rename legends as necessary
plt.figure(figsize=(10, 5))
plt.plot(time_grid, N_avg, label='Avg Fish Population (N)', color='blue')
plt.plot(time_grid, P_avg, label='Avg Fisher Population (P)', color='green')
plt.axhline(N_star, color='blue', linestyle='--', label='Equilibrium N*')
plt.axhline(P_star, color='green', linestyle='--', label='Equilibrium P*')
plt.xlabel('Time')
plt.ylabel('Population Density')
plt.title('Average of 100 Gillespie Simulations (CPU Parallel via joblib)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()