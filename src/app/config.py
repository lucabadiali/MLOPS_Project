import os
from enum import Enum
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


class ModelSource(str, Enum):
    HF = "hf"
    LOCAL = "local"

MODEL_SOURCE = ModelSource(os.getenv("MODEL_SOURCE", "hf"))
HF_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_model_and_tokenizer(MODEL_SOURCE):
    if MODEL_SOURCE == ModelSource.HF:   # use the latest model available in the HF hub
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    else: # use a locally fine tuned model
        local_model_path = Path("models/saved_model")
        assert local_model_path.exists(), """No local model was found. Run 'python3 src/train_model.py'"""
        tokenizer = AutoTokenizer.from_pretrained("models/saved_tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained("models/saved_model")
    return tokenizer, model
