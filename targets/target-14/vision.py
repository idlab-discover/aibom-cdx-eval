from transformers import AutoProcessor, AutoModelForVision2Seq

VISION_MODEL = "microsoft/Florence-2-base"
processor = AutoProcessor.from_pretrained(VISION_MODEL)
model = AutoModelForVision2Seq.from_pretrained(VISION_MODEL)
