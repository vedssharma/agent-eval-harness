FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest==9.2.0

LABEL agent_eval.image="base"
WORKDIR /workspace
