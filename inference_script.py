from setfit import SetFitModel
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = "setfit_sentiment_model_v1"

def load_model():
    print(f"Loading SetFit model from {MODEL_DIR}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the fine-tuned model
    model = SetFitModel.from_pretrained(MODEL_DIR)
    model.to(device)
    return model

def predict_sentiment(model, sentences):
    """
    Predicts sentiment labels for a list of sentences.
    """
    print(f"Predicting on {len(sentences)} sentences...")
    preds = model.predict(sentences)
    
    # Probas (if you need confidence scores)
    # Note: SetFit's predict_proba returns probabilities for all classes
    probas = model.predict_proba(sentences)
    
    results = []
    for i, sentence in enumerate(sentences):
        results.append({
            "text": sentence,
            "predicted_label": preds[i],
            "confidence_scores": probas[i].tolist() # Convert tensor to list
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # 1. Load
    model = load_model()
    
    # 2. Dummy Data (Xero context)
    new_data = [
        "This feature is absolutely critical for my accounting workflow, please add it!", # Likely V. Positive
        "The new update is okay, but it broke the invoice sorting.",                    # Mixed/Negative
        "I hate the new UI, it's confusing and slow.",                                  # V. Negative
        "Just a question about the API limits.",                                        # Neutral
        "Good job on the payroll fix."                                                  # Positive
    ]
    
    # 3. Predict
    df_results = predict_sentiment(model, new_data)
    
    print("\nResults:")
    print(df_results)