#!/bin/bash
# Download models for offline use
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google-bert/bert-base-cased')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openai-community/gpt2-medium')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('FacebookAI/roberta-base')"
