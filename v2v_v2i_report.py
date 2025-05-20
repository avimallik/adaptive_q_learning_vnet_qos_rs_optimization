import matplotlib.pyplot as plt
import numpy as np

# LRI bins as labels
lri_bins_labels = ['0.8–1.0', '0.6–0.8', '0.4–0.6', '0.2–0.4', '<0.2']

# Message reliability percentages
reliability_q_learning = [98.5, 95.1, 89.3, 78.6, 64.2]
reliability_baseline = [94.2, 89.0, 82.4, 67.5, 52.1]

x = np.arange(len(lri_bins_labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(9,5))
rects1 = ax.bar(x - width/2, reliability_q_learning, width, label='Q-Learning Model', color='blue')
rects2 = ax.bar(x + width/2, reliability_baseline, width, label='Baseline Static Model', color='red')

ax.set_xlabel('Link Reliability Index (LRI)')
ax.set_ylabel('Message Reliability (%)')
ax.set_title('Message Reliability vs. Wireless Link Quality')
ax.set_xticks(x)
ax.set_xticklabels(lri_bins_labels)
ax.set_ylim(40, 105)
ax.legend()
ax.grid(axis='y')

# Adding data labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
