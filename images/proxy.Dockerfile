FROM python:3.12-slim

RUN pip install --no-cache-dir aiohttp==3.14.5 pydantic==2.14.1

COPY src/agent_eval /app/agent_eval
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app
ENTRYPOINT ["python", "-m", "agent_eval.proxy"]
