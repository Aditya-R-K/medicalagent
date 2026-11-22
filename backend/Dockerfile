FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# tiktoken fetches its encoding file from the internet on first use. Baked in
# so passage sizes are measured the same way on every machine, online or not.
ENV TIKTOKEN_CACHE_DIR=/srv/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

RUN useradd --create-home --uid 10001 service \
    && mkdir -p /data/documents \
    && chown -R service:service /data /srv
USER service

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
