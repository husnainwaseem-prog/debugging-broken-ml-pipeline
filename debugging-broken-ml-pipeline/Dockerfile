FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs the full comparison (both pipelines + ablation + plot) and the tests.
CMD ["sh", "-c", "python -m src.cli evaluate && python -m pytest tests/ -q"]
