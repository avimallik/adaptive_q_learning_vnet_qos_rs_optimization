import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Simulation parameters
num_vehicles = 100
np.random.seed(42)

# Q-learning hyperparameters
num_actions = 4
alpha_q = 0.1
gamma_q = 0.9
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
num_episodes = 1000
epsilon = epsilon_start

# Synthetic data for 100 vehicles
delays = np.random.uniform(10, 100, num_vehicles)
pdrs = np.random.uniform(0.7, 1.0, num_vehicles)
trusts = np.random.uniform(0.5, 1.0, num_vehicles)
safety_flags = np.random.randint(0, 2, num_vehicles)
energies = np.random.uniform(0.1, 0.3, num_vehicles)

# Actions: predefined weight sets (α, β, γ, δ)
actions = [
    (0.25, 0.25, 0.25, 0.25),
    (0.1, 0.4, 0.4, 0.1),
    (0.4, 0.2, 0.2, 0.2),
    (0.2, 0.3, 0.3, 0.2)
]

# Enhanced state: delay + trust + safety
# 5 delay buckets * 3 trust levels * 2 safety flags = 30 states
def get_state_rich(delay, trust, safety):
    delay_bucket = min(int(delay / 20), 4)  # 0-4
    trust_bucket = min(int(trust / 0.2), 2)  # 0-2
    return delay_bucket * 6 + trust_bucket * 2 + safety  # 30 unique states

num_states_rich = 30
Q_table = np.zeros((num_states_rich, num_actions))

# Ground truth optimal actions (heuristic-based random for simulation)
true_optimal_actions_rich = np.random.choice(num_actions, num_states_rich)

# Tracking
tracking_rewards = []
tracking_accuracy = []

for episode in range(num_episodes):
    cumulative_reward = 0
    for i in range(num_vehicles):
        state = get_state_rich(delays[i], trusts[i], safety_flags[i])
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(Q_table[state])

        w_alpha, w_beta, w_gamma, w_delta = actions[action]
        qos = w_alpha * (1 / delays[i]) + w_beta * pdrs[i] + w_gamma * trusts[i] + w_delta * safety_flags[i]
        reward = 10 * qos - 2 * energies[i]
        cumulative_reward += reward

        # Normalize QoS as proxy for reward
        norm_reward = (qos - 0.2) / (1.0 - 0.2)

        next_state = state
        Q_table[state, action] += alpha_q * (norm_reward + gamma_q * np.max(Q_table[next_state]) - Q_table[state, action])

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_end)

    # Accuracy tracking
    predicted_actions = np.argmax(Q_table, axis=1)
    correct = [predicted_actions[s] == true_optimal_actions_rich[s] for s in range(num_states_rich)]
    accuracy = np.mean(correct)
    tracking_rewards.append(cumulative_reward)
    tracking_accuracy.append(accuracy)

# Final accuracy evaluation
final_predicted_actions = np.argmax(Q_table, axis=1)
final_accuracy = np.mean([final_predicted_actions[s] == true_optimal_actions_rich[s] for s in range(num_states_rich)]) * 100
print(f"Final Accuracy: {final_accuracy:.2f}%")

# Plotting convergence
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(tracking_rewards, label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Reward Convergence")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.array(tracking_accuracy) * 100, label="Accuracy (%)", color='orange')
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Convergence")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
