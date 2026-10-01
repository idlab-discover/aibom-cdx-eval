from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_dataset

TICKETS = load_dataset("acme-ml/support-tickets", split="train")
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
