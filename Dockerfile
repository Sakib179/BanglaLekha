FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch \
    OMP_NUM_THREADS=2 \
    MODEL_PATH=/app/models/resnet18.pt \
    VOWEL_HELPER_DIR=/app/models/vowel_helper

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-hybrid.txt requirements-segmentation.txt ./
# Install CPU wheels; CUDA is unnecessary for this server.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY app.py ocr_service.py hybrid_ocr.py segmentation_ocr.py labels.json ./
COPY ui/ ui/
COPY .streamlit/ .streamlit/
RUN useradd --create-home appuser && mkdir -p models && chown appuser:appuser models
USER appuser

# Mount local models/ read-only when running the container.
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
