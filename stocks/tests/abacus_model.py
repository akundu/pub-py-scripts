from abacusai import ApiClient

# Initialize client
client = ApiClient(api_key=="s2_7709c4235ed64f2aab529ae65bc0ddde")
# List available RouteLLM models
models = client.list_route_llm_models()

print("Available models:")
for model in models:
    print(f"  - {model}")
