from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
from langchain_openai import ChatOpenAI

embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
reranker_model = ChatOpenAI(model="BAAI/bge-reranker-v2-m3")
flag_model = FlagModel("BAAI/bge-reranker-v2-m3")
