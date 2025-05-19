import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
n = 100  # Number of vehicles

# Define message types and corresponding safety priority weights
priority_weights = {
    'infotainment': 0.0,
    'congestion': 0.5,
    'accident': 0.8,
    'ambulance': 1.0
}

# Generate synthetic dataset
vehicles = pd.DataFrame({
    'vehicle_id': [f'V{i+1}' for i in range(n)],
    'delay_ms': np.random.uniform(10, 100, n),             # End-to-end delay
    'pdr': np.random.uniform(0.7, 1.0, n),                  # Packet delivery ratio
    'trust_score': np.random.uniform(0.5, 1.0, n),          # Trust score
    'energy_j': np.random.uniform(0.1, 0.3, n),             # Energy consumption
    'msg_type': np.random.choice(
        list(priority_weights.keys()), n, 
        p=[0.6, 0.2, 0.1, 0.1]                              # Probability distribution
    )
})

# Assign safety score based on message type
vehicles['safety_score'] = vehicles['msg_type'].map(priority_weights)

# Define QoS model weights (e.g., suburban scenario)
alpha, beta, gamma, delta = 0.25, 0.25, 0.3, 0.2
lambda_, mu = 10, 2

# Compute QoS score
vehicles['qos_score'] = (
    alpha * (1 / vehicles['delay_ms']) +
    beta * vehicles['pdr'] +
    gamma * vehicles['trust_score'] +
    delta * vehicles['safety_score']
)

# Compute Reward
# vehicles['reward'] = lambda_ * vehicles['qos_score'] - mu * vehicles['energy_j']

# Save to CSV for analysis
vehicles.to_csv("vehicular_fog_simulation_output.csv", index=False)

# Preview first 10 rows
print(vehicles.head(10))