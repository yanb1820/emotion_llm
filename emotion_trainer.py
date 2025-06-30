from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

GOEMOTIONS_TO_EKMAN = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "disgust",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}
EKMAN_CATEGORIES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

def load_and_prepare_dataset():

    go_emotions_dataset = load_dataset("go_emotions")
    label_names = go_emotions_dataset["train"].features["labels"].feature.names

    #map original go emotions labels to ekman 7 labels for the data
    def map_labels(data_row):
        labels = {GOEMOTIONS_TO_EKMAN.get(label_names[i]) for i in data_row["labels"]}
        labels.discard(None)
        data_row["ekman_labels"] = list(labels)
        return data_row

    #filter the data and only keep single-label data
    def filter_zero_label(data_row):
        return len(data_row["ekman_labels"]) == 1

    go_emotions_dataset = go_emotions_dataset.map(map_labels)
    go_emotions_dataset = go_emotions_dataset.filter(filter_zero_label)
    return go_emotions_dataset

def tokenize_and_encode(dataset, tokenizer, label2id):
    def preprocess(row):
        encoding = tokenizer(row["text"].lower().strip(), padding="max_length", truncation=True, max_length=128)
        encoding["label"] = label2id[row["ekman_labels"][0]]
        return encoding

    encoded = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    encoded.set_format("torch")
    return encoded


# the way to measure model’s performance.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

dataset = load_and_prepare_dataset()
# loads the lowercase version of DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
label_to_id = {label: idx for idx, label in enumerate(EKMAN_CATEGORIES)}
encoded_dataset = tokenize_and_encode(dataset, tokenizer, label_to_id)

def train_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(EKMAN_CATEGORIES)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Save model after each epoch
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./final_emotion_model")  # saves weights, config, training setup
    tokenizer.save_pretrained("./final_emotion_model")  # saves vocab, rules, config
if __name__ == "__main__":
    train_model()