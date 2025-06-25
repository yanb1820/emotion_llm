import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from collections import Counter

goemotions_to_ekman = {
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

ekman_categories = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

go_emotions_dataset = load_dataset("go_emotions")

# Load label names
label_names = go_emotions_dataset["train"].features["labels"].feature.names
#check if it's correct, use count
all_labels = [label for data_row in go_emotions_dataset["train"] for label in data_row["labels"]]

#map original go emotions labels to ekman 7 labels for the data
def map_labels(data_row):
    mapped = set()
    for label_idx in data_row["labels"]:
        label_name = label_names[label_idx]
        ekman_label = goemotions_to_ekman.get(label_name)
        if ekman_label:
            mapped.add(ekman_label)
    data_row["ekman_labels"] = list(mapped)
    return data_row

#filter the data and only keep single-label data
def filter_zero_label(data_row):
    return len(data_row["ekman_labels"]) == 1

go_emotions_dataset = go_emotions_dataset.map(map_labels) #Hugging Face will pass in one data row at a time
go_emotions_dataset = go_emotions_dataset.filter(filter_zero_label) # filter 0 labels

#loads the lowercase version of DistilBERT, good for casual text like social media.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#creat a dictionary for ekman labels with id
ekman_to_id = {label: i for i, label in enumerate(ekman_categories)}
id_to_ekman= {id:ekman for ekman, id in ekman_to_id.items()}

def preprocess(data_row):
    # Clean the text: lowercase and remove spaces
    text = data_row["text"].lower().strip()
    # Get the first (and only) label from ekman_labels, then convert to number
    label = ekman_to_id[data_row["ekman_labels"][0]]

    # Tokenize text
    encoding = tokenizer(text,padding="max_length", truncation=True,max_length=128)
    encoding["label"] = label
    return encoding

#drop all original columns, only keep return from preprocess
encoded_dataset = go_emotions_dataset.map(preprocess, remove_columns=go_emotions_dataset["train"].column_names)
#Tell the Dataset to Work with PyTorch
encoded_dataset.set_format("torch")

#import the model
from transformers import AutoModelForSequenceClassification
# 7 classes = 6 basic emotions + neutral
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=7
)
# #freeze base model weights and training only classification head
# for param in model.distilbert.parameters():
#     param.requires_grad = False
#
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
# the way to measure model’s performance.
def compute_metrics(pred):
    #real answers
    labels = pred.label_ids
    #np.argmax(...) just picks the biggest number’s index: That’s what the model is guessing.
    preds = np.argmax(pred.predictions, axis=1)
    #compares real answers to predictions answers,  and calculates
    acc = accuracy_score(labels, preds)
    #checks how good the predictions are, if it's good even rare ones
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


from transformers import TrainingArguments, Trainer
if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="./results",            # Where to save model
        eval_strategy="epoch",       # Evaluate after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,  # Automatically load the best epoch
        metric_for_best_model="eval_accuracy",  # Best according to which number?
        greater_is_better=True, # Should that number go up or down to be better?
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./final_emotion_model") #saves weights, config, training setup
    tokenizer.save_pretrained("./final_emotion_model") #saves vocab, rules, config