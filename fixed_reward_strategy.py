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

# Fixed reward values for tiers
reward_map = {1: 10, 2: 6, 3: 3}  # Tier 1 > Tier 2 > Tier 3

# Assign tiers based on event flags
tiers = []
for i in range(num_vehicles):
    if ambulance_flags[i] == 1:
        tiers.append(1)
    elif crash_alert_flags[i] == 1:
        tiers.append(2)
    else:
        tiers.append(3)

# Fixed QoS weights (α, β, γ, δ)
fixed_weights = (0.25, 0.25, 0.25, 0.25)

qos_scores = []
reward_scores = []

for i in range(num_vehicles):
    safety_flag = int(ambulance_flags[i] or crash_alert_flags[i] or road_hazard_flags[i])
    qos = (fixed_weights[0] * (1 / delays[i]) +
           fixed_weights[1] * pdrs[i] +
           fixed_weights[2] * trusts[i] +
           fixed_weights[3] * safety_flag)
    reward = reward_map[tiers[i]]  # fixed reward by tier, no energy penalty here
    
    qos_scores.append(qos)
    reward_scores.append(reward)

# Create DataFrame
results_df = pd.DataFrame({
    'Vehicle': [f'V{i+1}' for i in range(num_vehicles)],
    'QoS Score': np.round(qos_scores, 3),
    'Reward Score': reward_scores
})

print(results_df)

# Plotting QoS Scores line graph
plt.figure(figsize=(10, 5))
plt.plot(results_df['Vehicle'], results_df['QoS Score'], marker='o', linestyle='-', color='blue', label='QoS Score')
plt.title('QoS Scores per Vehicle (Fixed Reward Strategy)')
plt.xlabel('Vehicle')
plt.ylabel('QoS Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting Reward Scores line graph
plt.figure(figsize=(10, 5))
plt.plot(results_df['Vehicle'], results_df['Reward Score'], marker='s', linestyle='-', color='green', label='Reward Score')
plt.title('Reward Scores per Vehicle (Fixed Reward Strategy)')
plt.xlabel('Vehicle')
plt.ylabel('Reward Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
