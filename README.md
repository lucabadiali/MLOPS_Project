---
title: Sentiment Analysis API
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

The top part of this file is needed to run a docker image in Hugging Face Docker Space (more details below).

# Sentiment Analysis API

This is a small project that allows the user to fine tune or download a pretrained Sentiment Analysis model taken from this [Hugging Face repo](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), and to then use this model via API to classify input texts into the following categories : positive, negative, and neutral.

## Project Deliverables Overview

- ✔ Sentiment analysis model using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- ✔ Fine-tuning script on public sentiment dataset (`tweet_eval`)
- ✔ CI pipeline (pytest + linting)
- ✔ CD pipeline deploying a Dockerized FastAPI app on HF Spaces
- ✔ Continuous monitoring with Prometheus + Grafana

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       ├── huggingface-space-deploy.yml
│       └── python-app.yml
├── data/
│   └── load_data.py
├── models/
├── src/
│   ├── app/
│   │   ├── __pycache__/
│   │   │   └── config.cpython-311.pyc
│   │   ├── __init__.py
│   │   ├── app_post.py
│   │   ├── app.py
│   │   ├── config.py
│   │   └── utils.py
│   └── train_model.py
├── tests/
│   └── test_app.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── env_config.sh
├── prometheus.yml
├── pytest.ini
├── README.md
└── requirements.txt
```

## Project Configuration

The file [env_config.sh](env_config.sh) defines the following environmental variables for the shell that launches the app:
- MODEL_SOURCE: "hf" or "local". It indicates whether the app should use a locally trained model ("hf") or an online published model ("local");
EVAL_SAMPLE_SIZE: how many samples to use for the app monitoring (see below for more details);
EVAL_PERIOD_MIN1: how often monitoring tasks should be run;
EVAL_BATCH_SIZE: batch size used when evaluating the model for monitoring tasks;
TRAIN_FRACTION_SIZE: fraction of the train dataset to use for training the model (the training can take some hours on my GPU and so I included the chance for a faster training)
EVAL_FRACTION_SIZE: Similar as above but just for the model evaluation during training.

## Train Model

The user can decide to fine tune a pretrained model by running the python script [src/train_model.p](src/train_model.py) . If no dataset is already present in the project folder, the script downloads the *tweet_eval* dataset for the *sentiment* task from the same HF repo. Once the script completes, a model will be saved within the *models* folder.

## API calls

The user can decide whether to use a locally fine tuned model or the latest available model in the HF space linked above; more on this and other app configurations is described below.
The FastAPI app implemented in src/app/app.py has a *predict* endpoint that can receive a list of input texts and for each text returns the sentiment prediction and the probability score for all possible sentiments.
For demonstration purposes the script *src/app/app_post.py* can be run to obtain some predictions.


## CI

The file *tests/test_app.py* implements some Pytest tests that check the response of the app to post requests. These tests are run automatically when pushing to the [project github repo](https://github.com/lucabadiali/MLOPS_Project/actions) using the github action defined in the file *MLOPS_Project/.github/workflows/python-app.yml*. This action additionally runs flake8 on the project for code linting.

## CD

The app is hosted at [this HF Docker Space](https://huggingface.co/spaces/lucabadiali/ML_OPS_Project). HF runs the docker image specified in the *Dockerfile* of the project folder. The *Dockerfile* tells the container to:
- install the required Python version;
- install the necessary packages listed in the file *requirements.txt*;
- load the configuration variables defined in the *env_config.sh* file and runs the app via *uvicorn*.

A second remote to the HF Space repo was added to my local repo, so that my local repo could push both to the github and HF Space repo. To ensure automatic synchronization between these two, I added the guthub action defined in *MLOPS_Project/.github/workflows/huggingface-space-deploy.yml*. This action, triggered every time my local repo pushes to the github repo, makes sure that the same exact project folder is also pushed to the HF Space repo. In order for this action to work I created an HF access token to my HF repo, which I then saved as github secret (see the yml file). 

Overall this automization makes such that every time some edits are pushed from my local to my github repo, HF consequently builds and hosts the most updated version of the app.

## MONITORING

A *Prometheus* image was set to scrape metrics from the HF hosted running app and to send those metrics to a *Grafana* image. Both these images were composed locally through the [docker-compose.yml](docker-compose.yml) file in the project folder. Specific settings for Prometheus, like the endpoint to scrape from, are specified in the [prometheus.yml](prometheus.yml) file.
The metrics collected by Prometheus can be read at https://lucabadiali-ml-ops-project.hf.space/metrics . 

Since no real-time labelled data stream is available, the monitoring loop 
uses randomly sampled labelled data from the test set to simulate incoming 
data. Specifically, a job scheduler periodically runs the following tasks:
- creation of a random subset of the test data and evaluation of the model accuracy;
- creation of a random subset of the test data, model prediction and computation of sentiment distribution.

Finally on my Grafana image I created a dashbord with:
- a time series visualization of the model accuracy;
- a time series visualization of the sentiment distribution;
- a piechart visualization of the latest sentiment distribution.

Snapshots of such panels can be found in the [snapshot](snapshots) folder.











