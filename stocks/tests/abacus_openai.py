from openai import OpenAI

# Initialize with Abacus RouteLLM endpoint
client = OpenAI(
    api_key="s2_7709c4235ed64f2aab529ae65bc0ddde",
    base_url="https://api.abacus.ai/api/v0/chat"  # Try this endpoint
)

# Make a simple test call
response = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[
        {"role": "user", "content": "What are the top 3 tech stocks to watch in 2025?"}
    ]
)

print(response.choices[0].message.content)
