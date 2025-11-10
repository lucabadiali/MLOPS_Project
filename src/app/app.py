from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import preprocess
from scipy.special import softmax
import numpy as np
from pydantic import BaseModel
import urllib.request
import csv
import requests
from typing import Union, List
import torch



app = FastAPI()


class SentimentQuery(BaseModel):
    input_texts: Union[str, List[str]]

task='sentiment'
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

MODEL = f"cardiffnlp/twitter-roberta-base-{task}-latest"
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)



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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
