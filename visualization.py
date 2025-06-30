import matplotlib.pyplot as plt
import numpy as np

# compare vader and fine-tune model by visualizing
classes = ["joy", "neutral", "sadness"]

vader_f1 = [0.73, 0.49, 0.56]
model_f1 = [0.82, 0.68, 0.69]

x = np.arange(len(classes))
width = 0.35  # width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vader_f1, width, label='VADER')
bars2 = ax.bar(x + width/2, model_f1, width, label='DistilBERT')

ax.set_ylabel('F1-score')
ax.set_title('F1-score by Emotion Class: VADER vs DistilBERT')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.show()