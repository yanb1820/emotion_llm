# Emotion Classification with DistilBERT (Ekman 7 Labels)

This project fine-tunes DistilBERT on the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset and maps the 28 original emotions to 6 Ekman emotions + neutral.


## 1. Label Mapping

GoEmotions → Ekman: joy, sadness, anger, fear, surprise, disgust, neutral.

## 2. How It Works

- Uses `transformers`, `datasets`, `scikit-learn`, and PyTorch
- Applies multi-label → single-label filtering
- Fine-tunes DistilBERT with Hugging Face Trainer API

## 3. Project Files Overview

- **`emotion_trainer.py`**  
  - Fine-tunes DistilBERT on the GoEmotions dataset, mapped to Ekman’s six emotions + neutral.  
  - Includes preprocessing, label mapping, tokenization, and training with Hugging Face `Trainer`.  
  - Saves model and tokenizer to `./final_emotion_model`.

- **`load_and_evaluate_model.py`**  
  - Loads the saved model from `./final_emotion_model`.  
  - Evaluates on the test set — prints accuracy and F1 score.  
  - Shows misclassified examples: text, true label, predicted label.

- **`gradio_app.py`**  
  - Runs a Gradio web app for real-time emotion prediction.  
  - Takes input, shows predicted emotion + confidence.  
  - Uses `transformers-interpret` to highlight important words.

- **`vader_baseline.py`**  
  - Runs sentiment analysis using VADER for comparison with BERT model.

- **`bertviz_visualize.ipynb`**  
  - Visualizes model attention with BERTViz to explore what the model focuses on.

- **`final_emotion_model/`**  
  - Contains the fine-tuned model and tokenizer saved from `emotion_trainer.py`.

- **`results/`**  
  - Contains training logs and metrics from Hugging Face `Trainer`.

## 4. Motivation

This project connects NLP with cognitive science by mapping emotions to Ekman’s basic emotion theory. It helps build explainable AI for chatbots, social media analysis, and educational tools.
## 5. Performance

After 3 epochs:

- Accuracy: ~71%
- Weighted F1: ~0.70

## 6. Quickstart

```bash
python emotion_trainer.py
python load_and_evaluate_model.py
python gradio_app.py