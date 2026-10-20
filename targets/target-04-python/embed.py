from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
from langchain_openai import ChatOpenAI

embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
reranker_model = ChatOpenAI(model="BAAI/bge-reranker-v2-m3")
flag_model = FlagModel("BAAI/bge-reranker-v2-m3")

from datasets import load_dataset

# Passages to index: a dataset repo published by the same org as the embedder
corpus = load_dataset("sentence-transformers/natural-questions", split="train")
