import numpy as np
import matplotlib.pyplot as plt

# Simulated delay samples (ms) for demonstration purposes
np.random.seed(42)
delay_baseline = np.random.normal(120.5, 30, 1000)
delay_qlearning = np.random.normal(78.3, 20, 1000)

# Emergency response time samples (ms)
ert_baseline = np.random.normal(250.7, 50, 1000)
ert_qlearning = np.random.normal(145.9, 30, 1000)

# Figure 1: Delay distribution histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(delay_baseline, bins=50, alpha=0.6, label='Baseline', color='red', density=True)
plt.hist(delay_qlearning, bins=50, alpha=0.6, label='Q-Learning', color='blue', density=True)
plt.xlabel('End-to-End Delay (ms)')
plt.ylabel('Normalized Frequency')
plt.title('Distribution of End-to-End Safety Message Delays')
plt.legend()
plt.grid(True)

# Figure 2: ERT cumulative distribution function
plt.subplot(1, 2, 2)
plt.hist(ert_baseline, bins=100, density=True, cumulative=True, alpha=0.6, label='Baseline', color='red')
plt.hist(ert_qlearning, bins=100, density=True, cumulative=True, alpha=0.6, label='Q-Learning', color='blue')
plt.xlabel('Emergency Response Time (ms)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Emergency Response Times')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
