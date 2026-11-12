# Imagen para el proceso de paper trading y el dashboard (sin torch: la inferencia usa LR + LightGBM).
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY docs ./docs
RUN pip install --no-cache-dir -e .
ENV QUANT_AI_MODE=paper PYTHONUNBUFFERED=1
# data/, models/ y state/ se montan como volúmenes (ver compose.yaml)
CMD ["python", "scripts/run_paper.py"]
