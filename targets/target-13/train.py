from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer

BASE_MODEL = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

trainer = Trainer(model=model)
trainer.train()
