import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed seed for reproducibility
np.random.seed(42)

# Synthetic data for 10 vehicles
num_vehicles = 10
delays = np.random.uniform(10, 100, num_vehicles)
pdrs = np.random.uniform(0.7, 1.0, num_vehicles)
trusts = np.random.uniform(0.5, 1.0, num_vehicles)
energies = np.random.uniform(0.1, 0.3, num_vehicles)
ambulance_flags = np.random.randint(0, 2, num_vehicles)
crash_alert_flags = np.random.randint(0, 2, num_vehicles)
road_hazard_flags = np.random.randint(0, 2, num_vehicles)

# Example optimized weights from Q-learning
optimized_weights = (0.2, 0.3, 0.3, 0.2)  # (α, β, γ, δ)
lambda_, mu = 10, 2

priority_scaler = 1 + ambulance_flags * 0.5 + crash_alert_flags * 0.3 + road_hazard_flags * 0.2

qos_scores = []
reward_scores = []

for i in range(num_vehicles):
    safety_flag = int(ambulance_flags[i] or crash_alert_flags[i] or road_hazard_flags[i])
    qos = (optimized_weights[0] * (1 / delays[i]) +
           optimized_weights[1] * pdrs[i] +
           optimized_weights[2] * trusts[i] +
           optimized_weights[3] * safety_flag)
    qos *= priority_scaler[i]
    reward = lambda_ * qos - mu * energies[i]
    
    qos_scores.append(qos)
    reward_scores.append(reward)

# Create DataFrame
results_df = pd.DataFrame({
    'Vehicle': [f'V{i+1}' for i in range(num_vehicles)],
    'Delay (ms)': delays.round(2),
    'PDR': pdrs.round(3),
    'Trust': trusts.round(3),
    'Ambulance': ambulance_flags,
    'Crash Alert': crash_alert_flags,
    'Road Hazard': road_hazard_flags,
    'QoS Score': np.round(qos_scores, 3),
    'Reward Score': np.round(reward_scores, 3)
})

print(results_df)

# Plotting QoS Scores
plt.figure(figsize=(10, 5))
plt.bar(results_df['Vehicle'], results_df['QoS Score'], color='blue')
plt.xlabel('Vehicle')
plt.ylabel('QoS Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plotting Reward Scores
plt.figure(figsize=(10, 5))
plt.bar(results_df['Vehicle'], results_df['Reward Score'], color='green')
plt.xlabel('Vehicle')
plt.ylabel('Reward Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
