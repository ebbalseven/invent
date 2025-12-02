# Hafif ve üretime uygun Python imajı
FROM python:3.9-slim

# ---------- Temel ayarlar ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=2 \
    HF_HOME=/app/.cache/huggingface

# Çalışma dizini
WORKDIR /app

# ---------- Sistem paketleri ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python bağımlılıkları ----------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Uygulama dosyaları ----------
COPY . .

# Non-root user (opsiyonel ama önerilir)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Port
EXPOSE 8000

# Healthcheck (opsiyonel, orkestrasyon için güzel)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Başlangıç komutu
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
