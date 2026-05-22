FROM python:3.11-slim
RUN pip install fastapi uvicorn transformers torch
COPY . /app
WORKDIR /app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
