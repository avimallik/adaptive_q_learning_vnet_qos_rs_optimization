import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
episodes = 2000
x = np.arange(episodes)

# Simulated accuracy and rewards over training
policy_accuracy_qaperf = 70 + 16 * np.tanh((x - 800) / 200)
policy_accuracy_qrsp = 68 + 6 * np.tanh((x - 1300) / 250)

reward_qaperf = 6 + 3 * np.tanh((x - 900) / 300)
reward_qrsp = 6 + 1.5 * np.tanh((x - 1200) / 400)

# Plotting
plt.figure(figsize=(12, 5))

# Policy Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(x, policy_accuracy_qaperf, label='Q-APERF', color='blue')
plt.plot(x, policy_accuracy_qrsp, label='Q-RSP (Baseline)', color='red')
plt.title('Policy Accuracy Over Training')
plt.xlabel('Episode')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Reward Plot
plt.subplot(1, 2, 2)
plt.plot(x, reward_qaperf, label='Q-APERF', color='blue')
plt.plot(x, reward_qrsp, label='Q-RSP (Baseline)', color='red')
plt.title('Average Reward Over Training')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.suptitle('Figure 7.12: Accuracy and Reward Comparison Between Q-APERF and Q-RSP', y=1.02)
plt.show()
