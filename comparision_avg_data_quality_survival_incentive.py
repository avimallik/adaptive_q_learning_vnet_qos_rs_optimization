import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

labels = ['FRI', 'QAI', 'SII (Proposed)']
avg_quality = [0.72, 0.85, 0.78]
avg_survival = [0.55, 0.50, 0.75]
avg_swdi = [0.22, 0.21, 0.44]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width, avg_quality, width, label='Average Data Quality')
rects2 = ax.bar(x, avg_survival, width, label='Average Survival')
rects3 = ax.bar(x + width, avg_swdi, width, label='Average SWDI')

ax.set_xlabel('Incentive Mechanism')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()
