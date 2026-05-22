from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NER_PIPELINE = pipeline("ner", model="dslim/bert-base-NER")

chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)

@app.post("/chat")
def chat(text: str):
    inputs = chat_tokenizer(text, return_tensors="pt")
    return chat_model.generate(**inputs)

@app.post("/ner")
def ner(text: str):
    return NER_PIPELINE(text)
