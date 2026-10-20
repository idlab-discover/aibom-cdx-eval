import { pipeline } from "@xenova/transformers";

const MODEL_ID = "Xenova/bert-base-NER";
const SENTIMENT_MODEL = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

async function runNER(text: string) {
  const ner = await pipeline("token-classification", MODEL_ID);
  return ner(text);
}

async function runSentiment(text: string) {
  const classifier = await pipeline("sentiment-analysis", SENTIMENT_MODEL);
  return classifier(text);
}
