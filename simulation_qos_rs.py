import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Provided parameters and data initialization
num_vehicles = 100
np.random.seed(42)

# Synthetic data regeneration
delays = np.random.uniform(10, 100, num_vehicles)
pdrs = np.random.uniform(0.7, 1.0, num_vehicles)
trusts = np.random.uniform(0.5, 1.0, num_vehicles)
energies = np.random.uniform(0.1, 0.3, num_vehicles)

# Explicit event flags: ambulance, crash alert, road hazard
ambulance_flags = np.random.randint(0, 2, num_vehicles)
crash_alert_flags = np.random.randint(0, 2, num_vehicles)
road_hazard_flags = np.random.randint(0, 2, num_vehicles)

# Actions: predefined weight sets (α, β, γ, δ)
actions = [
    (0.25, 0.25, 0.25, 0.25),
    (0.1, 0.4, 0.4, 0.1),
    (0.4, 0.2, 0.2, 0.2),
    (0.2, 0.3, 0.3, 0.2)
]

# Enhanced state function
def get_state_rich(delay, trust, safety):
    delay_bucket = min(int(delay / 20), 4)
    trust_bucket = min(int(trust / 0.2), 2)
    return delay_bucket * 6 + trust_bucket * 2 + safety

num_states_rich = 30
Q_table = np.zeros((num_states_rich, len(actions)))

# Final run to populate Q-table (simplified for demonstration)
for i in range(num_vehicles):
    safety_flag = max(ambulance_flags[i], crash_alert_flags[i], road_hazard_flags[i])
    state = get_state_rich(delays[i], trusts[i], safety_flag)
    action = np.argmax(Q_table[state])
    Q_table[state, action] += 1  # assume incrementally learned

# Get best actions from final Q-table
final_actions = np.argmax(Q_table, axis=1)

# Priority scaler based on events
priority_scaler = 1 + (ambulance_flags * 0.5) + (crash_alert_flags * 0.3) + (road_hazard_flags * 0.2)

# Compute final QoS and Reward per vehicle
qos_scores = np.zeros(num_vehicles)
reward_scores = np.zeros(num_vehicles)
lambda_, mu = 10, 2

for i in range(num_vehicles):
    safety_flag = max(ambulance_flags[i], crash_alert_flags[i], road_hazard_flags[i])
    state = get_state_rich(delays[i], trusts[i], safety_flag)
    w_alpha, w_beta, w_gamma, w_delta = actions[final_actions[state]]
    base_qos = w_alpha * (1 / delays[i]) + w_beta * pdrs[i] + w_gamma * trusts[i] + w_delta * safety_flag
    qos_scores[i] = base_qos * priority_scaler[i]
    reward_scores[i] = lambda_ * qos_scores[i] - mu * energies[i]

# Compile results
results_final_df = pd.DataFrame({
    "Vehicle": [f"V{i+1}" for i in range(num_vehicles)],
    "Delay (ms)": delays,
    "PDR": pdrs,
    "Trust": trusts,
    "Ambulance": ambulance_flags,
    "Crash Alert": crash_alert_flags,
    "Road Hazard": road_hazard_flags,
    "Energy (J)": energies,
    "QoS Score": qos_scores,
    "Reward Score": reward_scores
})

# Visualization QoS Scores
plt.figure(figsize=(12, 6))
plt.bar(results_final_df["Vehicle"], results_final_df["QoS Score"], color='blue')
plt.xlabel("Vehicle")
plt.ylabel("QoS Score")
plt.title("Final QoS Score per Vehicle")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization Reward Scores
plt.figure(figsize=(12, 6))
plt.bar(results_final_df["Vehicle"], results_final_df["Reward Score"], color='green')
plt.xlabel("Vehicle")
plt.ylabel("Reward Score")
plt.title("Final Reward Score per Vehicle")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

# Display results DataFrame
print(results_final_df[["Vehicle", "QoS Score", "Reward Score"]])
