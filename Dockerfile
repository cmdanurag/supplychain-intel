# 3.14 to match the interpreter the pinned versions were actually chosen for.
# Four dependencies had to be bumped to find 3.14 wheels; running the container
# on an older Python would put it on a combination nothing has ever tested.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# LightGBM and scikit-learn link the OpenMP runtime, which slim images omit.
# Without this the build succeeds and the container dies on import with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker caches the pip layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY forecasting ./forecasting
COPY models ./models

EXPOSE 8000

# $PORT is injected by Render/Railway; 8000 is the local default.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
