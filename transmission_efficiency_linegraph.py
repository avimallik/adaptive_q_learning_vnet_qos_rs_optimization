import matplotlib.pyplot as plt

# Packet Loss Rates (%)
plr = [5, 10, 20, 30, 40]

# Average Transmission Efficiency (%) for Q-learning and Baseline
ate_q_learning = [96.8, 94.2, 89.5, 82.1, 75.4]
ate_baseline = [94.3, 90.5, 83.1, 73.2, 62.8]

plt.figure(figsize=(8, 5))
plt.plot(plr, ate_q_learning, marker='o', linestyle='-', color='blue', label='Q-Learning Model')
plt.plot(plr, ate_baseline, marker='s', linestyle='--', color='red', label='Baseline Static QoS')
plt.xlabel('Packet Loss Rate (%)')
plt.ylabel('Average Transmission Efficiency (%)')
plt.title('Average Transmission Efficiency vs. Packet Loss Rate')
plt.grid(True)
plt.legend()
plt.ylim(50, 100)
plt.xticks(plr)
plt.tight_layout()
plt.show()
