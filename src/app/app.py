from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import preprocess, load_model_and_tokenizer
from scipy.special import softmax
import numpy as np
from pydantic import BaseModel
import urllib.request
import csv
import requests
from typing import Union, List
import torch
from .config import MODEL_SOURCE, ModelSource
from prometheus_fastapi_instrumentator import Instrumentator

##################
from prometheus_client import Counter, Gauge
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
import random
import pandas as pd
#################


#############
from .config import EVAL_BATCH_SIZE, N_SAMPLES, DATASET_PATH, EVAL_PERIOD_MIN
from .utils import load_dataset
###########

app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

###################
# ---------- Metrics (custom) ----------
# Production predictions distribution (unlabeled)
# PRED_COUNTER = Counter(
#     "sentiment_requests_total",
#     "Total predictions served by label",
#     ["label"]
# )



# EVAL_SAMPLE_SIZE = Gauge(
#     "model_evaluation_sample_size",
#     "Number of samples used in the latest periodic evaluation"
# )
# EVAL_COUNTER_DIST = Counter(
#     "sentiment_test_distribution_total",
#     "Cumulative predicted label counts on evaluation samples",
#     ["label"]
# )
# EVAL_RUNS = Counter(
#     "model_evaluations_total",
#     "Total number of evaluation runs completed"
# )
##################






class SentimentQuery(BaseModel):
    input_texts: Union[str, List[str]]

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

tokenizer, model = load_model_and_tokenizer(MODEL_SOURCE)
model.eval()

@app.get("/")
def read_root(): 
    return {"status": "ok", "message": "Sentiment API is running"}

@app.post("/predict")
async def analyze_text(query:SentimentQuery):

    if isinstance(query.input_texts, str):
        input_texts = [query.input_texts]
    else:  # already a List[str]
        input_texts = query.input_texts
    encoded_batch = tokenizer(
        [preprocess(t) for t in input_texts],
        padding=True,          # pad to same length
        truncation=True,       # truncate long texts
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model(**encoded_batch)
    
    logits = output[0].detach().cpu().numpy()
    scores = softmax(logits, axis=-1)
    pred_labels = scores.argmax(axis=-1)

    response_body = []
    for i,text in enumerate(input_texts):

        predicted = labels[pred_labels[i]]
        #PRED_COUNTER.labels(label=predicted).inc()


        response_body.append(
            {
                "input_text":text,
                "prediction":labels[pred_labels[i]],
                "scores":
                    {
                        "negative": float(scores[i][0]),
                        "neutral": float(scores[i][1]),
                        "positive": float(scores[i][2])
                    }
            })

    return {
        "status" : "successful",
        "response_body": response_body
    }



def evaluate_accuracy():
    dataset = load_dataset(DATASET_PATH).shuffle()["test"][:N_SAMPLES]
    N_BATCHES = len(dataset["text"])//EVAL_BATCH_SIZE

    accuracy = 0
    for i in range(N_BATCHES+1):
        if i == N_BATCHES :
            samples, labels = dataset["text"][i*EVAL_BATCH_SIZE:], dataset["label"][i*EVAL_BATCH_SIZE:]
        else:
            samples, labels = dataset["text"][i*EVAL_BATCH_SIZE:(i+1)*EVAL_BATCH_SIZE], dataset["label"][i*EVAL_BATCH_SIZE:(i+1)*EVAL_BATCH_SIZE]

        model.eval()
        encoded_batch = tokenizer(
            [preprocess(t) for t in samples],
            padding=True,          # pad to same length
            truncation=True,       # truncate long texts
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoded_batch)
    
        logits = output[0].detach().cpu().numpy()
        scores = softmax(logits, axis=-1)
        pred_labels = scores.argmax(axis=-1)
        accuracy += sum(pred_labels==labels)
    accuracy/=N_SAMPLES
    return accuracy


# Evaluation metrics (labeled test set)
EVAL_ACCURACY = Gauge(
    "model_evaluation_accuracy",
    "Accuracy on latest periodic evaluation of labeled test subset"
)

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import threading

_model_lock = threading.Lock()

def _run_eval_and_set_gauge():
    # If you expect concurrent requests to /predict, the lock prevents GPU/torch contention
    with _model_lock:
        acc = evaluate_accuracy()
    EVAL_ACCURACY.set(acc)


scheduler = BackgroundScheduler(daemon=True)

@app.on_event("startup")
def _start_scheduler():
    # run once soon after startup
    scheduler.add_job(_run_eval_and_set_gauge, next_run_time=datetime.now() + timedelta(seconds=2))
    # then every EVAL_PERIOD_MIN minutes
    scheduler.add_job(_run_eval_and_set_gauge, "interval", minutes=EVAL_PERIOD_MIN)
    scheduler.start()

@app.on_event("shutdown")
def _stop_scheduler():
    scheduler.shutdown(wait=False)


















if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
