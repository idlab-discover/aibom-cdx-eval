from transformers import AutoProcessor, AutoModelForVision2Seq
from langchain_community.llms import Ollama

VISION_MODEL = "microsoft/Florence-2-base"
processor = AutoProcessor.from_pretrained(VISION_MODEL)
model = Ollama(model=VISION_MODEL)
