import matplotlib.pyplot as plt

# Data
vehicles = [10, 20, 30, 40, 50]
q_aperf_time = [11.5, 12.0, 12.5, 12.5, 12.5]
dqn_heuristic_time = [22.0, 23.0, 24.0, 24.0, 24.0]
qrsp_time = [15.5, 16.5, 17.0, 17.5, 17.5]

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(vehicles, q_aperf_time, color='green', marker='o', label='Q-APERF')
plt.plot(vehicles, dqn_heuristic_time, color='blue', marker='o', label='DQN-Heuristic')
plt.plot(vehicles, qrsp_time, color='red', marker='o', label='Q-RSP')

# Labels and formatting
plt.xlabel('Number of Vehicles')
plt.ylabel('Training Time (Secs)')
plt.xticks(vehicles)
plt.ylim(10, 25)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
