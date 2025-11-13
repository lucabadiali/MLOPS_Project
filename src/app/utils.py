from pathlib import Path
from .config import ModelSource, HF_MODEL
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



def load_model_and_tokenizer(MODEL_SOURCE):
    if MODEL_SOURCE == ModelSource.HF:   # use the latest model available in the HF hub
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    else: # use a locally fine tuned model
        local_model_path = Path("models/saved_model")
        assert local_model_path.exists(), """No local model was found. Run 'python3 src/train_model.py' first"""
        tokenizer = AutoTokenizer.from_pretrained("models/saved_tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained("models/saved_model")
    return tokenizer, model


def load_dataset(dataset_path):
    if dataset_path.exists():
        dataset = load_from_disk(dataset_path)
    else:
        dataset = hf_load_dataset('tweet_eval', 'sentiment')
    return dataset