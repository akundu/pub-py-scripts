import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Get API key from environment
api_key = os.getenv("ABACUS_API_KEY")
if not api_key:
    raise ValueError("ABACUS_API_KEY environment variable not set")

# Initialize LangChain with Abacus.AI endpoint
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.abacus.ai/llm/v1",
    model="gpt-4o",  # or "gemini-pro", "claude-3-sonnet", etc.
    temperature=0.7
)

# Simple chat
response = llm.invoke([
    HumanMessage(content="Explain what Instacart does in 2 sentences.")
])

print(response.content)
