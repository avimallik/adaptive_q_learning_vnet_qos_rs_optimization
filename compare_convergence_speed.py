import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Data
vehicles = [10, 20, 30, 40, 50]
q_aperf = [1000, 1100, 1150, 1170, 1200]
dqn_heuristic = [1650, 1720, 1750, 1770, 1800]
qrsp = [1400, 1450, 1521, 1550, 1600]

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(vehicles, q_aperf, color='green', marker='o', label='Q-APERF')
plt.plot(vehicles, dqn_heuristic, color='blue', marker='o', label='DQN-Heuristic')
plt.plot(vehicles, qrsp, color='red', marker='o', label='Q-RSP')

# Labels and title
plt.xlabel('Number of Vehicles')
plt.ylabel('Convergence (Episodes)')
plt.xticks(vehicles)
plt.ylim(900, 1850)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()

# Show plot
plt.show()
