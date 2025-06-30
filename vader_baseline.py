from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from emotion_trainer import dataset

# Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# map Ekman labels to 3-class (joy/sadness/neutral)
EKMAN_TO_VALENCE = {
    "joy": "joy",
    "surprise": "joy",
    "anger": "sadness",
    "disgust": "sadness",
    "fear": "sadness",
    "sadness": "sadness",
    "neutral": "neutral"
}

# Ekman-like 3-class mapping
def vader_classify(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "joy"
    elif score <= -0.05:
        return "sadness"
    else:
        return "neutral"

def evaluate_vader_on_dataset():
    #use test set for evaluation
    texts = dataset["test"]["text"]
    true_labels = [row[0] for row in dataset["test"]["ekman_labels"]]  # assuming single label
    vader_preds = [vader_classify(text) for text in texts]

    mapped_true_labels = [EKMAN_TO_VALENCE[label] for label in true_labels]
    accuracy = accuracy_score(mapped_true_labels, vader_preds)

    print(f"VADER baseline accuracy: {accuracy:.4f}")
    print(classification_report(mapped_true_labels, vader_preds))

    #confusion matrix
    cm_vader = confusion_matrix(mapped_true_labels, vader_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_vader, display_labels=["joy", "neutral", "sadness"])
    disp.plot(cmap="Oranges", xticks_rotation=45)
    plt.title("Confusion Matrix - VADER Baseline")
    plt.show()


if __name__ == "__main__":
    evaluate_vader_on_dataset()