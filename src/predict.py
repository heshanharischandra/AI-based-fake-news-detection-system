import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = r"D:\FakeNewsResearch\models\final_xlmr_model"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)

label_map = {
    0: "FAKE",
    1: "REAL"
}


def detect_language(text):
    if any("\u0D80" <= ch <= "\u0DFF" for ch in text):
        return "Sinhala"
    return "English"


def predict_news(text):

    if not text.strip():
        return "No text entered", "-", "-"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][prediction_id].item()

    language = detect_language(text)
    prediction = label_map[prediction_id]
    confidence_text = f"{confidence * 100:.2f}%"

    return language, prediction, confidence_text


demo = gr.Interface(
    fn=predict_news,

    inputs=gr.Textbox(
        lines=12,
        label="Paste Sinhala or English News Text",
        placeholder="Paste full news article, Facebook post, or social media text here..."
    ),

    outputs=[
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],

    title="Sinhala & English Fake News Detection",
    description="Paste the complete news text and click Analyze."
)

demo.launch()