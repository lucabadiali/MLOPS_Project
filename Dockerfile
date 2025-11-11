# 1. Base Python image
FROM python:3.11-slim

# 2. Create a non-root user (HF likes this pattern)
RUN useradd -m -u 1000 user
USER user

# 3. Set working directory
ENV HOME=/home/user
WORKDIR /home/user/app

# 4. Copy dependency file and install deps
COPY --chown=user requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY --chown=user . .

# 6. Expose the port your app will listen on
EXPOSE 7860

# 7. Start the app
CMD ["python3" "-m" "uvicorn", "src.app.app:app", "--reload", "--host", "0.0.0.0", "--port", "7860"]
