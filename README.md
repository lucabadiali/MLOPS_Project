---
title: Sentiment Analysis API
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sentiment Analysis API

The top part of this file is meant to run a docker image in Hugging Face (more details below).

Structure of the project folder:

.github/
    └── workflows/
        ├── huggingface-space-deploy.yml
        └── python-app.yml
data/
    └── load_data.py
models/
src/
    ├── app/
        ├── __pycache__/
            └── config.cpython-311.pyc
        ├── __init__.py
        ├── app_post.py
        ├── app.py
        ├── config.py
        └── utils.py
    └── train_model.py
tests/
    └── test_app.py
.dockerignore
.gitignore
docker-compose.yml
Dockerfile
env_config.sh
prometheus.yml
pytest.ini
README.md
requirements.txt

