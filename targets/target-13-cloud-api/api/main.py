import os
from transformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(os.environ["LLM_MODEL"])
