FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY api ./api
COPY config ./config
COPY data ./data
COPY mcp_adapter ./mcp_adapter
COPY mcp_server.py ./
COPY model ./model
COPY observability ./observability
COPY prompts ./prompts
COPY rag ./rag
COPY safety ./safety
COPY services ./services
COPY utils ./utils

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir ".[production]"

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
