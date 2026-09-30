FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOLML_API_HOST=0.0.0.0 \
    MOLML_API_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md requirements-benchmark.txt ./
COPY src ./src

RUN python -m pip install --no-cache-dir --constraint requirements-benchmark.txt ".[api]" \
    && useradd --uid 10001 --create-home molml \
    && chown -R molml:molml /app /home/molml

USER molml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["molml-api"]
