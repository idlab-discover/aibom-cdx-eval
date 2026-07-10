FROM python:3.11-slim
ARG MODEL_ID=meta-llama/Llama-2-7b-chat-hf
ENV MODEL_ID=${MODEL_ID}
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RUN pip install transformers torch
COPY . /app
WORKDIR /app
CMD ["python", "serve.py"]
