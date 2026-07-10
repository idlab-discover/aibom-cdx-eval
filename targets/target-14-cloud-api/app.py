"""
Cloud AI chatbot using LangChain with OpenAI, Anthropic, and Ollama.
No local model weights — all inference is via API or a local Ollama server.
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# Primary chat model — OpenAI GPT-4o via LangChain
chat = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Cheaper model for simple / high-volume tasks
fast_chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

# Anthropic Claude for long-context summarisation
summarizer = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens=4096)

# Local Ollama model for privacy-sensitive requests (no data leaves the machine)
local_llm = Ollama(model="llama3.2")


def chat_with_model(model_choice: str, message: str) -> str:
    """Route a message to the appropriate backend model."""
    if model_choice == "gpt4o":
        return chat.predict(message)
    elif model_choice == "claude":
        return summarizer.predict(message)
    elif model_choice == "local":
        return local_llm(message)
    # Default to fast (cheap) OpenAI model
    return fast_chat.predict(message)
