import json
cfg = json.load(open("config.json"))
from transformers import AutoModel
model = AutoModel.from_pretrained(cfg["model"])
