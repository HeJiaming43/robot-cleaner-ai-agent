FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs data/tickets

EXPOSE 8501

# 健康检查只做本地校验，不触发模型调用，避免依赖抖动导致容器反复重启
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python scripts/healthcheck.py || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
