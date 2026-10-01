/**
 * Inference using @huggingface/transformers (browser/Node.js).
 */

import { pipeline, AutoModel, AutoTokenizer, env } from "@huggingface/transformers";
import { HfInference } from "@huggingface/inference";

// ── pipeline() ───────────────────────────────────────────────────────────────

const classifier = await pipeline("text-classification", "distilbert/distilbert-base-uncased-finetuned-sst-2-english");

const generator = await pipeline("text-generation", "Xenova/gpt2", { dtype: "q4" });


// ── .from_pretrained() ───────────────────────────────────────────────────────

const model = await AutoModel.from_pretrained("Xenova/bert-base-uncased");
const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

// Quantised model
const qModel = await AutoModel.from_pretrained(
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
);

// ── @huggingface/inference ────────────────────────────────────────────────────

const hf = new HfInference(process.env.HF_TOKEN);

const out1 = await hf.textGeneration({
  model: "meta-llama/Llama-3.1-8B-Instruct",
  inputs: "Once upon a time",
});

const out2 = await hf.imageClassification({
  model: "google/vit-base-patch16-224",
  data: imageBlob,
});

// ── Model chosen through a variable ─────────────────────────────────────────

const MODEL = "facebook/opt-1.3b";
const dynamicModel = await AutoModel.from_pretrained(MODEL);

const org = "Xenova";
const name = "bert-base-uncased";
const tlModel = await AutoModel.from_pretrained(`${org}/${name}`);

const useSmall = true;
const chosenModel = useSmall ? "Xenova/gpt2" : "Xenova/llama-3-8b";
const ternaryModel = await AutoModel.from_pretrained(chosenModel);
