from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from main import go_emotions_dataset,ekman_categories,encoded_dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Ekman-like 3-class mapping
def vader_classify(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "joy"        # Simplified positive
    elif score <= -0.05:
        return "sadness"    # Simplified negative
    else:
        return "neutral"

# Use test set for evaluation
texts = go_emotions_dataset["test"]["text"]
true_labels = [row[0] for row in go_emotions_dataset["test"]["ekman_labels"]]  # assuming single label

# Apply VADER
vader_preds = [vader_classify(text) for text in texts]



# Map your true Ekman labels to 3-class (joy/sadness/neutral)
ekman_to_valence = {
    "joy": "joy",
    "surprise": "joy",   # optional, or map to neutral
    "anger": "sadness",
    "disgust": "sadness",
    "fear": "sadness",
    "sadness": "sadness",
    "neutral": "neutral"
}

mapped_true_labels = [ekman_to_valence[label] for label in true_labels]

accuracy = accuracy_score(mapped_true_labels, vader_preds)
print(f"VADER baseline accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report



print(classification_report(mapped_true_labels, vader_preds))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_vader = confusion_matrix(mapped_true_labels, vader_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_vader, display_labels=["joy", "neutral", "sadness"])
disp.plot(cmap="Oranges", xticks_rotation=45)
plt.title("Confusion Matrix - VADER Baseline")
plt.show()

wrong = [(i, p, l) for i, (p, l) in enumerate(zip(mapped_true_labels, vader_preds)) if p != l]
print(f"Found {len(wrong)} VADER misclassified examples")