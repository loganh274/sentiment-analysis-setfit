import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from setfit import SetFitModel, SetFitTrainer, Trainer, TrainingArguments
from datasets import Dataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH = "xpi_labeled_data_augmented.csv"
MODEL_OUTPUT_DIR = "setfit_sentiment_model_v1"

# Better base model for sentiment nuances than MiniLM-L6
# "all-mpnet-base-v2" is widely considered the best general-purpose model 
# that fits easily on a standard GPU (approx 420MB).
BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 

# Training Params
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """
    Generates a confusion matrix to evaluate 5-point scale accuracy.
    """
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

def main():
    print("--- Starting SetFit Sentiment Model Training ---")

    # 1. HARDWARE CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Check: Training on {device.upper()}")

    # 2. LOAD DATA
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Identify columns (Adjust if your CSV headers are different)
    text_col = next((col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()), 'text')
    label_col = next((col for col in df.columns if 'label' in col.lower() or 'sentiment' in col.lower()), 'label')
    
    # Clean data
    df = df.dropna(subset=[text_col, label_col])
    
    # IMPORTANT: Ensure labels are consistent strings if needed
    # If your labels are 1-5, keep them as is. If strings "Very Positive", ensure consistency.
    unique_labels = sorted(df[label_col].unique())
    print(f"Found {len(unique_labels)} unique classes: {unique_labels}")

    # Split Data (Train/Test)
    # We use a standard split to validate if the model generalizes
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    # Convert to Hugging Face Dataset format (required by SetFit)
    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]])
    test_ds = Dataset.from_pandas(test_df[[text_col, label_col]])

    # 3. INITIALIZE SETFIT MODEL
    print(f"Loading base model: {BASE_MODEL_NAME}")
    # SetFitModel.from_pretrained loads the SentenceTransformer body + a Classification head
    model = SetFitModel.from_pretrained(
        BASE_MODEL_NAME,
        labels=unique_labels,
        multi_target_strategy=None # Use default for multi-class
    )
    
    # Move to GPU
    model.to(device)

    # 4. TRAIN (FINE-TUNE)
    # This process:
    # A. Generates sentence pairs (positive/negative pairs) from your data
    # B. Fine-tunes the embedding body using Contrastive Loss (making same-sentiment closer, different further)
    # C. Trains a Logistic Regression head on the new embeddings
    print("Starting training...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class="CosineSimilarityLoss",
        metric="accuracy",
        batch_size=BATCH_SIZE,
        num_iterations=20, # Number of pairs to generate per sentence
        num_epochs=NUM_EPOCHS,
        column_mapping={text_col: "text", label_col: "label"}
    )
    
    trainer.train()

    # 5. EVALUATION
    print("Evaluating on Test Set...")
    metrics = trainer.evaluate()
    print(f"Test Metrics: {metrics}")
    
    # Detailed Report
    preds = model.predict(test_df[text_col].tolist())
    print("\nClassification Report:")
    print(classification_report(test_df[label_col], preds))
    
    # Plot Confusion Matrix
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    plot_confusion_matrix(test_df[label_col], preds, unique_labels, MODEL_OUTPUT_DIR)

    # 6. SAVE MODEL
    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    # This saves the full fine-tuned transformer + the classifier head
    model.save_pretrained(MODEL_OUTPUT_DIR)
    
    print("Done! Model is ready for Snowflake deployment.")

if __name__ == "__main__":
    main()