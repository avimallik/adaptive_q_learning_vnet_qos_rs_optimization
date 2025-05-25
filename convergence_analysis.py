import matplotlib.pyplot as plt
import numpy as np

# Assume tracking_rewards and tracking_accuracy are collected from your Q-learning training
# Example arrays (replace with your actual training logs)
episodes = np.arange(1, 1001)
tracking_rewards = np.linspace(100, 800, 1000) + np.random.normal(0, 20, 1000)  # simulated reward increase
tracking_accuracy = np.linspace(0.3, 0.83, 1000)  # simulated accuracy increase

plt.figure(figsize=(12, 5))

# Plot cumulative reward
plt.subplot(1, 2, 1)
plt.plot(episodes, tracking_rewards, label="Cumulative Reward", color='blue')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.legend()

# Plot policy accuracy
plt.subplot(1, 2, 2)
plt.plot(episodes, tracking_accuracy * 100, label="Policy Accuracy (%)", color='orange')
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
