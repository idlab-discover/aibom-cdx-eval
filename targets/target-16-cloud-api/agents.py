"""
Multi-agent orchestration: planner (OpenAI) + critic (Anthropic) + coder (Ollama local).
Models are specified directly in API call arguments — common pattern with direct SDK clients.
"""

from anthropic import Anthropic
from openai import OpenAI
from langchain_community.llms import Ollama

# Direct SDK clients (no LangChain wrapper)
openai_client = OpenAI()
anthropic_client = Anthropic()

# Local model for code generation (privacy / cost)
code_llm = Ollama(model="codellama")


def run_planner(task: str) -> str:
    """High-level task planning via GPT-4o."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": task}],
    )
    return response.choices[0].message.content


def run_critic(draft: str) -> str:
    """Review and critique a draft answer via Claude."""
    message = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": f"Review this draft and suggest improvements:\n\n{draft}"}],
        max_tokens=1024,
    )
    return message.content[0].text


def run_coder(spec: str) -> str:
    """Generate code locally using Ollama CodeLlama (no data sent to cloud)."""
    return code_llm(f"Write Python code for: {spec}")
