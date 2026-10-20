import json
import os

import boto3
from sentence_transformers import SentenceTransformer

bedrock = boto3.client("bedrock-runtime")
embedder = SentenceTransformer(os.environ["EMBED_MODEL"])


def summarize(text: str) -> str:
    response = bedrock.invoke_model(
        modelId=os.environ["BEDROCK_MODEL_ID"],
        body=json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 512}),
    )
    return json.loads(response["body"].read())["content"][0]["text"]


def embed(texts: list[str]):
    return embedder.encode(texts)
