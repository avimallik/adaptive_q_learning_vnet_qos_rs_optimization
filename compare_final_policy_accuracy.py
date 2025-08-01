import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Vehicle counts
vehicles = [10, 20, 30, 40, 50]

# Final Policy Accuracy data
q_aperf_accuracy = [85.0, 87.0, 89.0, 92.0, 95.5]
dqn_heuristic_accuracy = [82.0, 84.0, 85.5, 86.8, 87.8]
qrsp_accuracy = [78.0, 80.0, 82.0, 83.5, 84.6]

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(vehicles, q_aperf_accuracy, color='green', marker='o', label='Q-APERF')
plt.plot(vehicles, dqn_heuristic_accuracy, color='blue', marker='o', label='DQN-Heuristic')
plt.plot(vehicles, qrsp_accuracy, color='red', marker='o', label='Q-RSP')

# Labels and title
plt.xlabel('Number of Vehicles')
plt.ylabel('Policy Accuracy (%)')
plt.xticks(vehicles)
plt.ylim(75, 100)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

# Display the plot
plt.show()
