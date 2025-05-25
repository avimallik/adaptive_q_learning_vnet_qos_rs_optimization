import matplotlib.pyplot as plt

# Average latency (ms)
latency = [45.3, 54.7, 63.8, 77.4, 92.1]

# Average energy consumption (Joules)
energy = [0.28, 0.24, 0.21, 0.18, 0.15]

# Labels for different weight configurations
labels = ['(1.0, 0.0)', '(0.75, 0.25)', '(0.5, 0.5)', '(0.25, 0.75)', '(0.0, 1.0)']

plt.figure(figsize=(8,5))
plt.plot(latency, energy, marker='o', linestyle='-', color='purple')
for i, label in enumerate(labels):
    plt.annotate(label, (latency[i], energy[i]), textcoords="offset points", xytext=(5,-10), ha='center')
plt.xlabel('Average Latency (ms)')
plt.ylabel('Average Energy Consumption (Joules)')
plt.grid(True)
plt.tight_layout()
plt.show()
