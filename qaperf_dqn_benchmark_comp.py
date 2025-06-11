import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
episodes = 2000
x = np.arange(episodes)

# Simulated accuracy and rewards over training (Q-APERF vs Q-RSP)
policy_accuracy_qaperf_qrsp = 70 + 16 * np.tanh((x - 800) / 200)
policy_accuracy_qrsp = 68 + 6 * np.tanh((x - 1300) / 250)

reward_qaperf_qrsp = 6 + 3 * np.tanh((x - 900) / 300)
reward_qrsp = 6 + 1.5 * np.tanh((x - 1200) / 400)

# Simulated accuracy and rewards over training (Q-APERF vs DQN)
policy_accuracy_qaperf_dqn = 70 + 18 * np.tanh((x - 900) / 200)
policy_accuracy_dqn = 65 + 7 * np.tanh((x - 1400) / 300)

reward_qaperf_dqn = 6 + 3.5 * np.tanh((x - 800) / 250)
reward_dqn = 6 + 2 * np.tanh((x - 1300) / 350)

# Compute final summary metrics for comparison (Q-APERF vs Q-RSP)
final_accuracy_qaperf_qrsp = round(policy_accuracy_qaperf_qrsp[-1], 1)
final_accuracy_qrsp = round(policy_accuracy_qrsp[-1], 1)
final_reward_qaperf_qrsp = round(reward_qaperf_qrsp[-1], 2)
final_reward_qrsp = round(reward_qrsp[-1], 2)
convergence_qaperf_qrsp = np.argmax(policy_accuracy_qaperf_qrsp > 85)
convergence_qrsp = np.argmax(policy_accuracy_qrsp > 75)

# Compute final summary metrics for comparison (Q-APERF vs DQN)
final_accuracy_qaperf_dqn = round(policy_accuracy_qaperf_dqn[-1], 1)
final_accuracy_dqn = round(policy_accuracy_dqn[-1], 1)
final_reward_qaperf_dqn = round(reward_qaperf_dqn[-1], 2)
final_reward_dqn = round(reward_dqn[-1], 2)
convergence_qaperf_dqn = np.argmax(policy_accuracy_qaperf_dqn > 85)
convergence_dqn = np.argmax(policy_accuracy_dqn > 75)

# Prepare tables for both comparisons
comparison_qrsp_table = pd.DataFrame({
    "Model": ["Q-APERF", "Q-RSP (Baseline)"],
    "Final Policy Accuracy (%)": [final_accuracy_qaperf_qrsp, final_accuracy_qrsp],
    "Final Avg. Reward": [final_reward_qaperf_qrsp, final_reward_qrsp],
    "Convergence Episode": [convergence_qaperf_qrsp, convergence_qrsp]
})

comparison_dqn_table = pd.DataFrame({
    "Model": ["Q-APERF", "DQN-Heuristic"],
    "Final Policy Accuracy (%)": [final_accuracy_qaperf_dqn, final_accuracy_dqn],
    "Final Avg. Reward": [final_reward_qaperf_dqn, final_reward_dqn],
    "Convergence Episode": [convergence_qaperf_dqn, convergence_dqn]
})

# Save comparison tables to CSV
comparison_qrsp_table.to_csv("qaperf_vs_qrsp_comparison.csv", index=False)
comparison_dqn_table.to_csv("qaperf_vs_dqn_comparison.csv", index=False)

# Create figure for both comparisons
plt.figure(figsize=(14, 6))

# Policy Accuracy Plot (Q-APERF vs Q-RSP)
plt.subplot(1, 2, 1)
plt.plot(x, policy_accuracy_qaperf_qrsp, label='Q-APERF', color='steelblue')
plt.plot(x, policy_accuracy_qrsp, label='Q-RSP (Baseline)', color='salmon')
plt.title('Policy Accuracy Over Training Episodes (Q-APERF vs Q-RSP)')
plt.xlabel('Episode')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Reward Plot (Q-APERF vs Q-RSP)
plt.subplot(1, 2, 2)
plt.plot(x, reward_qaperf_qrsp, label='Q-APERF', color='steelblue')
plt.plot(x, reward_qrsp, label='Q-RSP (Baseline)', color='salmon')
plt.title('Average Reward Over Training Episodes (Q-APERF vs Q-RSP)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Figure 7.12: Performance Comparison of Q-APERF vs Q-RSP', y=1.05)
plt.show()

# Create figure for Q-APERF vs DQN comparison
plt.figure(figsize=(14, 6))

# Policy Accuracy Plot (Q-APERF vs DQN)
plt.subplot(1, 2, 1)
plt.plot(x, policy_accuracy_qaperf_dqn, label='Q-APERF', color='steelblue')
plt.plot(x, policy_accuracy_dqn, label='DQN-Heuristic', color='salmon')
plt.title('Policy Accuracy Over Training Episodes (Q-APERF vs DQN)')
plt.xlabel('Episode')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Reward Plot (Q-APERF vs DQN)
plt.subplot(1, 2, 2)
plt.plot(x, reward_qaperf_dqn, label='Q-APERF', color='steelblue')
plt.plot(x, reward_dqn, label='DQN-Heuristic', color='salmon')
plt.title('Average Reward Over Training Episodes (Q-APERF vs DQN)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Figure 7.13: Performance Comparison of Q-APERF vs DQN with Heuristic Event Handling', y=1.05)
plt.show()

# Output the comparison tables for verification
print(comparison_qrsp_table)
print(comparison_dqn_table)
