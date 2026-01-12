from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="setfit_sentiment_model_safetensors", repo_id="loganh274/nlp-testing-setfit", repo_type="model")
