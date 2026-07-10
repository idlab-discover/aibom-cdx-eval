import os
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model=os.environ["CLASSIFIER_MODEL"])
