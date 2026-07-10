import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_ID"])
tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"])
