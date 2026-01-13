"""SetFit sentiment model training script."""

import pandas as pd
import numpy as np
import torch
import os
import json
import sys
import sklearn
import setfit
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss

# Configuration
TRAIN_DATA_PATH = "data/training.csv"
TEST_DATA_PATH = "data/test.csv"
MODEL_OUTPUT_DIR = "models/setfit_sentiment_model_safetensors"
CONFUSION_MATRIX_OUTPUT_DIR = "output/model_visualizations"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
#BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 16
NUM_EPOCHS = (1, 16)

def enhance_emotional_features(text):
    """Preserve and normalize emotional language features for better embedding."""
    # Normalize repeated punctuation (!!!! -> !! [EMPHASIS])
    text = re.sub(r'!{2,}', '!! [EMPHASIS]', text)
    text = re.sub(r'\?{2,}', '?? [QUESTION_EMPHASIS]', text)
    
    # Normalize repeated letters (sooooo -> so [ELONGATED])
    text = re.sub(r'(.)\1{2,}', r'\1\1 [ELONGATED]', text)
    
    # Mark ALL CAPS words (but don't lowercase them)
    def mark_caps(match):
        word = match.group(0)
        if len(word) > 2 and word.isupper():
            return f"{word} [CAPS]"
        return word
    text = re.sub(r'\b[A-Z]{2,}\b', mark_caps, text)
    
    return text

def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('SetFit Sentiment Classification Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    plt.close()


def save_deployment_metadata(output_dir, unique_labels):
    """Save environment and model metadata for deployment reproducibility."""
    metadata = {
        "labels": [int(l) if isinstance(l, (np.integer, int)) else str(l) for l in unique_labels],
        "environment": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "setfit": setfit.__version__,
            "torch": torch.__version__
        },
        "base_model": BASE_MODEL_NAME,
        "serialization": "safetensors"
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Deployment metadata saved to {output_dir}/model_metadata.json")


def get_col_names(df):
    """Detect text and label column names."""
    text = next((c for c in df.columns if 'text' in c.lower() or 'comment' in c.lower()), 'text')
    label = next((c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower()), 'label')
    return text, label


def main():
    print("--- Starting SetFit Sentiment Model Training ---")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  
    else:
        device = "cpu"
    print(f"Training on {device.upper()}")

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found at {TRAIN_DATA_PATH}")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")

    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    text_col, label_col = get_col_names(train_df)

    print("Enhancing emotional features in text...")
    train_df[text_col] = train_df[text_col].apply(enhance_emotional_features)
    test_df[text_col] = test_df[text_col].apply(enhance_emotional_features)
   
    train_df = train_df.dropna(subset=[text_col, label_col])
    test_df = test_df.dropna(subset=[text_col, label_col])

    unique_labels = sorted(train_df[label_col].unique().tolist())
    print(f"Training on {len(train_df)} rows. Found classes: {unique_labels}")
    print(f"Testing on {len(test_df)} rows.")

    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]])
    test_ds = Dataset.from_pandas(test_df[[text_col, label_col]])

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = SetFitModel.from_pretrained(BASE_MODEL_NAME, labels=unique_labels)
    model.to(device)

    print("Starting training...")
    args = TrainingArguments(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_iterations=10,
        loss=CosineSimilarityLoss,
        metric_for_best_model="embedding_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
        column_mapping={text_col: "text", label_col: "label"}
    )
    
    trainer.train()

    print("Evaluating on test set...")
    metrics = trainer.evaluate()
    print(f"Test Metrics: {metrics}")
    
    preds = model.predict(test_df[text_col].tolist())
    
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    plot_confusion_matrix(test_df[label_col], preds, unique_labels, CONFUSION_MATRIX_OUTPUT_DIR)

    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR, safe_serialization=True)
    save_deployment_metadata(MODEL_OUTPUT_DIR, unique_labels)
    print("Done! Model saved in safetensors format.")


if __name__ == "__main__":
    main()