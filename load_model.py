import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from main import encoded_dataset, compute_metrics, id_to_ekman, ekman_categories, go_emotions_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("final_emotion_model", num_labels=7)
tokenizer = AutoTokenizer.from_pretrained("final_emotion_model")

# Recreate the Trainer
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    eval_dataset=encoded_dataset["test"],
)

# Evaluate
results = trainer.evaluate()
predictions = trainer.predict(encoded_dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
print(results)


# for i in range(5):
#     print("Predicted:", id_to_ekman[preds[i]])
#     print("Actual   :", id_to_ekman[labels[i]])
#     print()

# cm = confusion_matrix(labels, preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ekman_categories)
# disp.plot(cmap="Blues", xticks_rotation=45)
# plt.title("Confusion Matrix - Emotion Classifier")
# plt.show()

wrong = [(i, p, l) for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
print(f"Found {len(wrong)} misclassified examples")

for idx, pred, true in wrong[:10]:
    text = go_emotions_dataset["test"][idx]["text"]
    true_label = go_emotions_dataset["test"][idx]["ekman_labels"][0]
#     print(f"\nText: {text}")
#     print(f"True: {true_label}, Predicted: {ekman_categories[pred]}")