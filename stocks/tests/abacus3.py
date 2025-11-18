from abacusai import ApiClient

# Initialize client with your API key
client = ApiClient(api_key='YOUR_API_KEY_HERE')

# Test a simple question
response = client.chat_llm(
    messages=[
        {"role": "user", "content": "What are the top 3 tech stocks to watch in 2025?"}
    ],
    model="claude-sonnet-4"
)

print(response)
