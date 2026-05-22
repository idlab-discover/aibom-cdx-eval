from transformers import AutoModelForCausalLM, AutoTokenizer
from model_registry import REGISTRY

def load_chat():
    model_id = REGISTRY["chat"]
    return AutoModelForCausalLM.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id)
