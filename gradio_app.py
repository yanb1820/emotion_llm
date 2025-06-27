from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr
from main import id_to_ekman
from transformers_interpret import SequenceClassificationExplainer
import numpy as np

model_path = "final_emotion_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=7)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
id_to_label = id_to_ekman

#for  transformations interpret
explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)

def predict_emotion(text):
    word_attributions = explainer(text)
    #remove [CLS] and [SEP], format output for display
    clean_attributions = [(word, score) for word, score in word_attributions if word not in ["[CLS]", "[SEP]"]]
    explanation = " \n".join([f" • {word} ({round(score, 2)})" for word, score in clean_attributions])

    pred_label_id = int(explainer.predicted_class_name.replace("LABEL_", ""))
    pred_label = id_to_ekman[pred_label_id]

    # compute confidence score 0-1
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    confidence = round(probs[0][pred_label_id].item(), 3)


    #normalize scores for coloring
    scores = np.array([score for _, score in clean_attributions])
    max_score = np.max(np.abs(scores)) or 1  # avoid div/0
    normalized = [score / max_score for score in scores]

    #HTML
    highlighted = ""
    for (word, score) in zip([w for w, _ in clean_attributions], normalized):
        opacity = abs(score)
        color = "rgba(255, 0, 0, {:.2f})".format(opacity) if score > 0 else "rgba(0, 0, 255, {:.2f})".format(opacity)
        highlighted += f"<span style='background-color: {color}'>{word} </span>"

    html_output = (
        f"<b>Predicted Emotion:</b> <span style='font-size:20px; color:#ff6666'>{pred_label}</span> "
        f"(Confidence: {confidence})<br><br>"
        f"<b>Attention Map:</b><br>"
        f"<div style='font-size:18px'>{highlighted}</div>"
    )

    return html_output, explanation


iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type something emotional..."),
    outputs=[gr.HTML(), gr.Textbox(label = "Each word contribution: ")],
    title="Emotion Classifier with Attention",
    description="This app predicts emotion with confidence score and highlights the important words"
)

iface.launch()