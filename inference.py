import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import mlflow

MODEL_DIR = "./model"
LABEL_NAMES = ["negative", "neutral", "positive"]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=3
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    
    return {
        "text": text,
        "sentiment": LABEL_NAMES[prediction],
        "confidence": round(max(probabilities) * 100, 2),
        "scores": {
            label: round(prob * 100, 2)
            for label, prob in zip(LABEL_NAMES, probabilities)
        }
    }

if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded successfully!")
    
    test_sentences = [
        "The company reported record profits this quarter.",
        "Stock prices fell sharply amid market uncertainty.",
        "The merger is expected to close by end of year.",
        "Investors are concerned about rising inflation.",
        "The startup raised $50 million in Series B funding."
    ]
    
    print("\n" + "="*60)
    print("FINANCIAL SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    for sentence in test_sentences:
        result = predict(sentence, tokenizer, model)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Scores: {result['scores']}")
        print("-"*60)