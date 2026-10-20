from transformers import AutoModel, AutoTokenizer
from langchain_openai import ChatOpenAI


def load_models():
    m1 = ChatOpenAI(model="WiebeVandendriessche/model-card-example")
    t1 = AutoModel.from_pretrained("templates/model-card-example")

    return m1, t1


if __name__ == "__main__":
    load_models()
