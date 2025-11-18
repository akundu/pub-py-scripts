from abacusai import ApiClient

client = ApiClient(api_key="s2_7709c4235ed64f2aab529ae65bc0ddde")

# # Use the internal _call_api method
# response = client._call_api(
#     'listRouteLLMModels',
#     'GET'
# )
#
# print(response)
#

# Ask a question using RouteLLM
question = "What are the top 3 tech stocks to watch in 2025 and why?"

try:
    response = client._call_api(
        'chatLLM',
        'POST',
        query_params={
            'message': question,
            'llmName': 'claude-sonnet-4'  # or any model from the list
        }
    )

    print(f"Question: {question}\n")
    print(f"Answer: {response.get('content', response)}")

except Exception as e:
    print(f"Error: {e}")
