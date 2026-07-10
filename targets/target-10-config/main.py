import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ["HF_MODEL_ID"]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
