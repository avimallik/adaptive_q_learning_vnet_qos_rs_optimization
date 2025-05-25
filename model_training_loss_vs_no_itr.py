import matplotlib.pyplot as plt
import numpy as np

# Simulated loss values for demonstration
iterations = np.arange(1, 2001)
loss_values = 0.9 * np.exp(-iterations / 700) + 0.05 * np.random.randn(2000)  # Exponential decay + noise
loss_values = np.clip(loss_values, a_min=0, a_max=None)  # No negative loss

plt.figure(figsize=(10,5))
plt.plot(iterations, loss_values, color='blue', label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
