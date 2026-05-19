# Sample Inference Predictions

## Model: DistilBERT fine-tuned on Financial News Sentiment

### Test Results

| Text | Sentiment | Confidence |
|---|---|---|
| Stock prices fell sharply amid market uncertainty | NEGATIVE | 96.1% |
| The merger is expected to close by end of year | POSITIVE | 95.01% |
| The startup raised $50 million in Series B funding | POSITIVE | 80.34% |
| Investors are concerned about rising inflation | NEGATIVE | 61.1% |
| The company reported record profits this quarter | NEUTRAL | 81.52% |

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 85.02% |
| F1 Score | 84.95% |
| Eval Loss | 0.4053 |
| Training Samples | 7,634 |
| Eval Samples | 1,909 |

---

## Training Progress

| Epoch | Accuracy | F1 Score |
|---|---|---|
| 1 | 82.56% | 82.14% |
| 2 | 83.71% | 83.96% |
| 3 | 85.02% | 84.95% |

## LoRA Configuration
- r = 16
- alpha = 32
- dropout = 0.1
- trainable params = 887,811 (1.3% of total)