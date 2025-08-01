import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Data
vehicles = [10, 20, 30, 40, 50]
q_aperf_violation = [1.1, 1.2, 1.3, 1.4, 1.3]
dqn_heuristic_violation = [4.2, 4.4, 4.7, 4.8, 5.5]
qrsp_violation = [6.8, 7.0, 7.2, 7.5, 7.8]

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(vehicles, q_aperf_violation, color='green', marker='o', label='Q-APERF')
plt.plot(vehicles, dqn_heuristic_violation, color='blue', marker='o', label='DQN-Heuristic')
plt.plot(vehicles, qrsp_violation, color='red', marker='o', label='Q-RSP')

# Labels and title
plt.xlabel('Number of Vehicles')
plt.ylabel('Priority Violation Rate (%)')
plt.xticks(vehicles)
plt.ylim(1, 8.5)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
