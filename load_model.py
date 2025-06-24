from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from main import encoded_dataset
from main import compute_metrics

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
print(results)

