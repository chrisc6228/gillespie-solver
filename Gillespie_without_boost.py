import numpy as np
import matplotlib.pyplot as plt

# Change model parameters based on your equation
r = 1.0      
K = 100       
a = 0.01      
s = 1.0       
k = 0.1       
N0 = 50     
d = 0.2      

Omega = 100   # Scale factor to reduce/increase stochasticity

#If you have a reference equilibrium, replace this with your values
#Otherwise this could be removed or commented out
N_star = N0 + np.arctanh(d / s) / k
P_star = (r / a) * (1 - N_star / K)

# Runs a single instance of Gillespie
def gillespie_scaled(N0_val, P0_val, T_max, dt_sample=0.1):
    np.random.seed()

    N = int(N0_val * Omega)
    P = int(P0_val * Omega)
    t = 0

    times = [0.0]
    N_series = [N / Omega]
    P_series = [P / Omega]
    next_sample_time = dt_sample

    while t < T_max and N > 0 and P > 0:
        # You can rename variables for your on model
        fish_birth = r * N * (1 - N / (K * Omega))
        fish_death = a * N * P / Omega
        tanh_term = np.tanh(k * (N / Omega - N0))
        fisher_birth = max(P * s * tanh_term, 0.0)
        fisher_death = d * P

        a0 = fish_birth + fish_death + fisher_birth + fisher_death
        if a0 <= 0:
            break # Can't have negatives

        # Uses the direct version of the algorithm
        dt = np.random.exponential(1 / a0)
        t += dt

        # Determine which reaction occurs
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

    return np.array(times), np.array(N_series), np.array(P_series)

# Run different number of simulations
num_runs = 100
T_max = 100
dt_sample = 1.0
time_grid = np.arange(0, T_max + dt_sample, dt_sample)
N_matrix = []
P_matrix = []

for _ in range(num_runs):
    _, N_vals, P_vals = gillespie_scaled(N0_val=0.6, P0_val=0.2, T_max=T_max, dt_sample=dt_sample)
    if len(N_vals) < len(time_grid):
        N_vals = np.pad(N_vals, (0, len(time_grid) - len(N_vals)), constant_values=N_vals[-1])
        P_vals = np.pad(P_vals, (0, len(time_grid) - len(P_vals)), constant_values=P_vals[-1])
    N_matrix.append(N_vals)
    P_matrix.append(P_vals)

N_avg = np.mean(N_matrix, axis=0)
P_avg = np.mean(P_matrix, axis=0)

# Rename Legend if necessary
plt.figure(figsize=(10, 5))
plt.plot(time_grid, N_avg, label='Avg Fish Population (N)', color='blue')
plt.plot(time_grid, P_avg, label='Avg Fisher Population (P)', color='green')
plt.axhline(N_star, color='blue', linestyle='--', label='Equilibrium N*')
plt.axhline(P_star, color='green', linestyle='--', label='Equilibrium P*')
plt.xlabel('Time')
plt.ylabel('Population Density')
plt.title('Average Trajectories over 100 Gillespie Simulations (Scaled Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
