import matplotlib.pyplot as plt

# Vehicle counts
vehicles = [10, 20, 30, 40, 50]

# Final Average Reward data
q_aperf_reward = [6.5, 7.2, 8.0, 8.9, 9.7]
dqn_heuristic_reward = [5.9, 6.4, 7.1, 8.0, 8.5]
qrsp_reward = [5.1, 5.6, 6.3, 7.1, 7.8]

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(vehicles, q_aperf_reward, color='green', marker='o', label='Q-APERF')
plt.plot(vehicles, dqn_heuristic_reward, color='blue', marker='o', label='DQN-Heuristic')
plt.plot(vehicles, qrsp_reward, color='red', marker='o', label='Q-RSP')

# Labels and title
plt.xlabel('Number of Vehicles')
plt.ylabel('Final Average Reward')
plt.xticks(vehicles)
plt.ylim(4.5, 10)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

# Display the plot
plt.show()
