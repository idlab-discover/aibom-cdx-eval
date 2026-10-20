from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFaceHub


summarizer = ChatAnthropic(model="facebook/bart-large-cnn")
extractor = HuggingFaceHub(model_name="openai/whisper-base")


if __name__ == "__main__":
    print("target-03-python synthetic fixture")
