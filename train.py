import os
import mlflow
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
OUTPUT_DIR = "./model"
NUM_LABELS = 3
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 128

LABEL_NAMES = ["negative", "neutral", "positive"]

def load_and_prepare_data():
    print("Loading financial sentiment dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )
    dataset = dataset.rename_column("text", "sentence")
    print(f"Dataset loaded: {dataset}")
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1": f1
    }

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("Financial Sentiment Fine-tuning")

    with mlflow.start_run(run_name="distilbert-lora-financial"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "lora_r": 16,
            "lora_alpha": 32
        })

        # Load data
        dataset = load_and_prepare_data()

        # Split dataset
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        # Load tokenizer and model
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label={i: label for i, label in enumerate(LABEL_NAMES)},
            label2id={label: i for i, label in enumerate(LABEL_NAMES)}
        )

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False
            )

        train_tok = train_dataset.map(tokenize_fn, batched=True)
        eval_tok = eval_dataset.map(tokenize_fn, batched=True)

        train_tok = train_tok.rename_column("label", "labels")
        eval_tok = eval_tok.rename_column("label", "labels")
        train_tok = train_tok.remove_columns(["sentence"])
        eval_tok = eval_tok.remove_columns(["sentence"])
        train_tok.set_format("torch")
        eval_tok.set_format("torch")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            report_to="none"
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print("Starting training...")
        trainer.train()

        # Evaluate
        print("Evaluating model...")
        results = trainer.evaluate()
        print(f"Final Results: {results}")

        mlflow.log_metrics({
            "final_accuracy": results["eval_accuracy"],
            "final_f1": results["eval_f1"],
            "eval_loss": results["eval_loss"]
        })

        # Save model
        print("Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        mlflow.log_artifact(OUTPUT_DIR)

        print(f"Training complete!")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()