from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub


def load_models():
    m1 = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    m2 = AutoModel.from_pretrained("facebook/opt-1.3b")
    m3 = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    m4 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    m5 = AutoTokenizer.from_pretrained("EMBO/vicreg_our")
    m6 = pipeline("text-generation", model="almanach/Gaperon-1125-24B-SFT")
    m7 = HuggingFaceEndpoint(repo_id="google/medgemma-4b-it")
    m8 = InferenceClient(model="google/medgemma-27b-text-it")
    m9 = HuggingFaceHub(repo_id="inceptionai/jais-adapted-70b")
    m10 = AutoModelForCausalLM.from_pretrained("Aleph-Alpha/llama-3_1-70b-tfree-hat-sft")

    return m1, m2, m3, m4, m5, m6, m7, m8, m9, m10


if __name__ == "__main__":
    load_models()
