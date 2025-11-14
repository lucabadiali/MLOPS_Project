###### IMPORTS 

########
# Imports for app and model creation and 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Union, List

##########
# Imports for model creation/usage
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import urllib.request
import csv

# #################
# LOCAL IMPORTS
from .config import MODEL_SOURCE, ModelSource, EVAL_BATCH_SIZE, EVAL_SAMPLE_SIZE, DATASET_PATH, EVAL_PERIOD_MIN
from .utils import preprocess, load_model_and_tokenizer, load_dataset

##################
# Imports for app monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import threading
#################



#################
# App creation and metrics exposition
app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)





#################
# class for transferring post request data
class SentimentQuery(BaseModel):
    input_texts: Union[str, List[str]]


#################
# Retrieve model either locally or via download
tokenizer, model = load_model_and_tokenizer(MODEL_SOURCE)
model.eval()

##############
# retrieve label to int mapping from model repo
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
#############


@app.get("/")
def read_root(): 
    return {"status": "ok", "message": "Sentiment API is running"}

@app.post("/predict")
async def analyze_text(query:SentimentQuery)->dict:
    """
    Elaborates an input query containing one or more text messages and returns a response
    containing the prediction and the sentiment score for each message
    """

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



# Evaluation metrics on labeled test set
EVAL_ACCURACY = Gauge(
    "model_evaluation_accuracy",
    "Accuracy on latest periodic evaluation of labeled test subset"
)

def evaluate_accuracy(N_SAMPLES:int, BATCH_SIZE:int)->float:
    """
    Evaluates and returns the model accuracy on a random subset of the test dataset 
    """
    dataset = load_dataset(DATASET_PATH).shuffle()["test"][:N_SAMPLES]
    N_BATCHES = len(dataset["text"])//BATCH_SIZE

    accuracy = 0
    for i in range(N_BATCHES+1):
        if i == N_BATCHES :
            samples, labels = dataset["text"][i*BATCH_SIZE:], dataset["label"][i*BATCH_SIZE:]
        else:
            samples, labels = dataset["text"][i*BATCH_SIZE:(i+1)*BATCH_SIZE], dataset["label"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]

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


# Sentiment Distribution over unlabelled set
SENTIMENT_BATCH_FRACTION = Gauge(
    "sentiment_batch_fraction",
    "Fraction of predictions in the latest monitored batch, by label (0..1).",
    ["label"]
)

def evaluate_sentiment_distribution(N_SAMPLES:int, BATCH_SIZE:int)->np.ndarray:
    """
    Evaluates and returns the sentiment distribution over a random subset of the test dataset
    """
    dataset = load_dataset(DATASET_PATH).shuffle()["test"][:N_SAMPLES]
    N_BATCHES = len(dataset["text"])//BATCH_SIZE

    model.eval()

    counts = np.array([0.,0.,0.])
    for i in range(N_BATCHES+1):
        if i == N_BATCHES :
            samples = dataset["text"][i*BATCH_SIZE:]
        else:
            samples = dataset["text"][i*BATCH_SIZE:(i+1)*BATCH_SIZE]

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
        counts += np.unique(pred_labels, return_counts=True)[1]

    fractions=counts/N_SAMPLES
    return fractions


##################
# scheduler creation for managing the metric creation jobs
scheduler = BackgroundScheduler(daemon=True)
# threading lock to possibly handle concurrent request
_model_lock = threading.Lock()


############
# jobs to be launched periodically

def _run_eval_and_send_data():
    with _model_lock:
        acc = evaluate_accuracy(EVAL_SAMPLE_SIZE, EVAL_BATCH_SIZE)
    EVAL_ACCURACY.set(acc)

def _run_sentiment_distr_and_send_data():
    with _model_lock:
        fractions = evaluate_sentiment_distribution(EVAL_SAMPLE_SIZE, EVAL_BATCH_SIZE)
    for i, label in enumerate(labels):
        SENTIMENT_BATCH_FRACTION.labels(label=label).set(fractions[i])


@app.on_event("startup")
def _start_scheduler():

    # run once soon after startup
    scheduler.add_job(_run_eval_and_send_data, next_run_time=datetime.now() + timedelta(seconds=2))
    # then every EVAL_PERIOD_MIN minutes
    scheduler.add_job(_run_eval_and_send_data, "interval", minutes=EVAL_PERIOD_MIN)
    
    # run once soon after startup 
    scheduler.add_job(_run_sentiment_distr_and_send_data, next_run_time=datetime.now() + timedelta(seconds=2))
    # then every EVAL_PERIOD_MIN minutes  
    scheduler.add_job(_run_sentiment_distr_and_send_data, "interval", minutes=EVAL_PERIOD_MIN)
    
    scheduler.start()

@app.on_event("shutdown")
def _stop_scheduler():
    scheduler.shutdown(wait=False)


















if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
