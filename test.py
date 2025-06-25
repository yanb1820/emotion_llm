import matplotlib.pyplot as plt
import numpy as np
# from vader_baseline import ekman_to_valence
# from main import id_to_ekman
# from load_model import preds, labels
#
#
#
#
#
# true_ekman_labels = [id_to_ekman[i] for i in labels]
# pred_ekman_labels = [id_to_ekman[i] for i in preds]
# true_valence = [ekman_to_valence[label] for label in true_ekman_labels]
# pred_valence = [ekman_to_valence[label] for label in pred_ekman_labels]
# from sklearn.metrics import classification_report
#
# print(classification_report(true_valence, pred_valence, target_names=["joy", "neutral", "sadness"]))


# Labels
classes = ["joy", "neutral", "sadness"]

# F1 scores
vader_f1 = [0.73, 0.49, 0.56]
model_f1 = [0.82, 0.68, 0.69]  # replace with your actual model's 3-class F1s if you calculate them

x = np.arange(len(classes))
width = 0.35  # width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vader_f1, width, label='VADER')
bars2 = ax.bar(x + width/2, model_f1, width, label='DistilBERT')

# Add labels, title, legend
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