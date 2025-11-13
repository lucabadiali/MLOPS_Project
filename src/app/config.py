import os
from enum import Enum
from pathlib import Path


class ModelSource(str, Enum):
    HF = "hf"
    LOCAL = "local"

MODEL_SOURCE = ModelSource(os.getenv("MODEL_SOURCE", "hf"))
HF_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = Path("data/dataset")


EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "80"))
EVAL_INTERVAL_HOURS = float(os.getenv("EVAL_INTERVAL_HOURS", "1"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

EVAL_BATCH_SIZE = 64
N_SAMPLES = 500
EVAL_PERIOD_MIN = 1


# def load_model_and_tokenizer(MODEL_SOURCE):
#     if MODEL_SOURCE == ModelSource.HF:   # use the latest model available in the HF hub
#         tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
#         model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
#     else: # use a locally fine tuned model
#         local_model_path = Path("models/saved_model")
#         assert local_model_path.exists(), """No local model was found. Run 'python3 src/train_model.py' first"""
#         tokenizer = AutoTokenizer.from_pretrained("models/saved_tokenizer")
#         model = AutoModelForSequenceClassification.from_pretrained("models/saved_model")
#     return tokenizer, model
