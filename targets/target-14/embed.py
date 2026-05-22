from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel

embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
reranker_model = "BAAI/bge-reranker-v2-m3"
flag_model = FlagModel(reranker_model)
