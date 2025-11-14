import os
from enum import Enum
from pathlib import Path

class ModelSource(str, Enum):
    HF = "hf"
    LOCAL = "local"

MODEL_SOURCE = ModelSource(os.getenv("MODEL_SOURCE", "hf"))
HF_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = Path("data/dataset")
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "100"))
EVAL_PERIOD_MIN = float(os.getenv("EVAL_PERIOD_MIN", "30"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "64"))
TRAIN_FRACTION_SIZE = float(os.getenv("TRAIN_FRACTION_SIZE", "0.2"))
EVAL_FRACTION_SIZE = float(os.getenv("EVAL_FRACTION_SIZE", "0.4"))

