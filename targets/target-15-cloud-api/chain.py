"""
Hybrid RAG pipeline: local HuggingFace embeddings + reranker, cloud LLM generation.

Pattern coverage:
  python_literal (from_pretrained / SentenceTransformer)  → aibomgen-cli, jorisbot, trusera
  python_literal (ChatOpenAI / ChatAnthropic class call)  → akumar-aibom, jorisbot, trusera
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Knowledge base to retrieve from
knowledge_base = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# ── Local open-weight models (HuggingFace) ────────────────────────────────────
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Cross-encoder for reranking retrieved passages
reranker_tokenizer = AutoTokenizer.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Cloud LLM for generation ──────────────────────────────────────────────────
# Primary: GPT-4o for complex reasoning
llm = ChatOpenAI(model_name="gpt-4o")

# Fallback: Anthropic Claude for long documents (200k context window)
fallback_llm = ChatAnthropic(model="claude-3-haiku-20240307")


def retrieve_and_generate(query: str, documents: list[str]) -> str:
    """Embed query, rerank docs, generate answer with cloud LLM."""
    # 1. Embed query
    query_embedding = embed_model.encode(query)

    # 2. Rerank retrieved passages (local, private)
    # ... reranking logic omitted for brevity ...

    # 3. Generate answer via cloud model
    context = "\n".join(documents[:3])
    return llm.predict(f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {query}")
