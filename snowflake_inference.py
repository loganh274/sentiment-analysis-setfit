"""
Snowflake ML Functions inference script for SetFit sentiment model.
Downloads the pretrained model from Hugging Face Hub.
"""
import os
from setfit import SetFitModel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_ID = "loganh274/nlp-testing-setfit"  # Your public Hugging Face model
CACHE_DIR = "/tmp/hf_cache"  # Snowflake-writable directory


# ---------------------------------------------------------------------------
# MODEL LOADING (Singleton pattern for efficiency)
# ---------------------------------------------------------------------------
_model = None

def get_model():
    """
    Loads and caches the model. Using a singleton pattern avoids 
    re-downloading the model on every function call in Snowflake.
    """
    global _model
    if _model is None:
        # Set cache directories for Hugging Face
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
        
        print(f"Downloading model from Hugging Face: {MODEL_ID}")
        _model = SetFitModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
        print("Model loaded successfully!")
    return _model


# ---------------------------------------------------------------------------
# INFERENCE FUNCTIONS (Use these in Snowflake UDFs)
# ---------------------------------------------------------------------------
def predict_sentiment(text: str) -> int:
    """
    Predicts sentiment for a single text string.
    Returns the predicted label as an integer.
    
    Usage in Snowflake:
        SELECT predict_sentiment(comment_text) FROM my_table;
    """
    model = get_model()
    prediction = model.predict([text])
    return int(prediction[0])


def predict_sentiment_batch(texts: list) -> list:
    """
    Predicts sentiment for a batch of texts.
    More efficient for processing multiple rows.
    
    Returns a list of predicted labels.
    """
    model = get_model()
    predictions = model.predict(texts)
    return [int(p) for p in predictions]


def predict_with_confidence(text: str) -> dict:
    """
    Predicts sentiment with confidence scores.
    
    Returns:
        dict with 'label' and 'confidence_scores'
    """
    model = get_model()
    prediction = model.predict([text])
    probas = model.predict_proba([text])
    
    return {
        "label": int(prediction[0]),
        "confidence_scores": probas[0].tolist()
    }


# ---------------------------------------------------------------------------
# LOCAL TESTING
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test locally before deploying to Snowflake
    test_texts = [
        "This feature is absolutely critical for my accounting workflow!",
        "The new update is okay, but it broke the invoice sorting.",
        "I hate the new UI, it's confusing and slow.",
    ]
    
    print("Testing model download and inference...")
    for text in test_texts:
        result = predict_with_confidence(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence_scores']}")
