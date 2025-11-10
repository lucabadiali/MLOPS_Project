from datasets import load_dataset, DatasetDict
from pathlib import Path


DATA_FOLDER_PATH = Path(__file__).resolve().parent
dataset_path = DATA_FOLDER_PATH / "dataset"

# def get_tweet_eval_sentiment() -> DatasetDict:
#     return load_dataset("tweet_eval", "sentiment")
dataset = load_dataset("tweet_eval", "sentiment")
dataset.save_to_disk(dataset_path)
