import matplotlib.pyplot as plt

plr = [5, 10, 20, 30, 40]
ate_q_learning = [96.8, 94.2, 89.5, 82.1, 75.4]
ate_baseline = [94.3, 90.5, 83.1, 73.2, 62.8]

plt.figure(figsize=(8, 5))
plt.scatter(plr, ate_q_learning, color='blue', s=100, label='Q-Learning Model')
plt.scatter(plr, ate_baseline, color='red', s=100, label='Baseline Static QoS')

plt.xlabel('Packet Loss Rate (%)')
plt.ylabel('Average Transmission Efficiency (%)')
plt.title('Average Transmission Efficiency vs. Packet Loss Rate')
plt.xticks(plr)
plt.ylim(50, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
