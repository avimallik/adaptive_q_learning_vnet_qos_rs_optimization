import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Network sizes
vehicle_counts = [100, 300, 500, 700, 1000]

# Completion rates (%)
tcr_q_learning = [97.2, 95.1, 92.8, 89.4, 85.7]
tcr_baseline = [93.4, 88.7, 84.5, 79.6, 73.2]

plt.figure(figsize=(8,5))
plt.plot(vehicle_counts, tcr_q_learning, marker='o', color='blue', label='Q-APERF')
plt.plot(vehicle_counts, tcr_baseline, marker='s', color='green', label='Baseline Static QoS')
plt.xlabel('Number of Vehicles')
plt.ylabel('Task/Message Completion Rate (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
