import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from emotion_trainer import (
    dataset,
    encoded_dataset,
    compute_metrics,
    EKMAN_CATEGORIES
)



def load_model_and_tokenizer(model_path="final_emotion_model"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=7)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def create_trainer(model, tokenizer):
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        eval_dataset=encoded_dataset["test"],
    )

def evaluate_and_display_results(trainer):
    results = trainer.evaluate()
    print("Evaluation Results:", results)

    predictions = trainer.predict(encoded_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    return preds, labels

def plot_confusion_matrix(labels, preds, show_plot=True):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EKMAN_CATEGORIES)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Emotion Classifier")
    plt.show()

def analyze_errors(preds, labels, show_text=True):
    wrong = [(i, p, l) for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    print(f"\nFound {len(wrong)} misclassified examples.\n")

    for idx, pred, true in wrong[:10]:
        text = dataset["test"][idx]["text"]
        print(f"Text: {text}")
        print(f"True: {EKMAN_CATEGORIES[true]}, Predicted: {EKMAN_CATEGORIES[pred]}\n")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    trainer = create_trainer(model, tokenizer)
    preds, labels = evaluate_and_display_results(trainer)

    #confusion matrix
    plot_confusion_matrix(labels, preds, show_plot=True)
    #run error analysis
    analyze_errors(preds, labels, show_text=False)