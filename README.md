# Financial News Sentiment Analysis — Fine-tuned DistilBERT with LoRA/PEFT

Fine-tuned DistilBERT on financial news sentiment classification using 
LoRA/PEFT for parameter-efficient training, achieving 85% accuracy on 
9,543 financial news samples.

## Results

| Metric | Score |
|---|---|
| Accuracy | 85.02% |
| F1 Score | 84.95% |
| Training Samples | 7,634 |
| Eval Samples | 1,909 |

## Tech Stack

- **Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning**: LoRA/PEFT (only 1.3% trainable parameters)
- **Dataset**: Twitter Financial News Sentiment (9,543 samples)
- **Tracking**: MLflow experiment tracking
- **Classes**: Negative, Neutral, Positive

## Sample Predictions

| Text | Sentiment | Confidence |
|---|---|---|
| Stock prices fell sharply amid market uncertainty | NEGATIVE | 96.1% |
| The merger is expected to close by end of year | POSITIVE | 95.01% |
| The startup raised $50 million in Series B funding | POSITIVE | 80.34% |

## Setup

1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install: `pip install transformers datasets peft torch accelerate mlflow scikit-learn`
5. Train: `python train.py`
6. Test: `python inference.py`
