# Lead qualification demo. Data and the recorded model run ship inside the image.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.32" "google-genai>=1.0"
COPY app.py ./
COPY data/ ./data/
COPY static/ ./static/
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
