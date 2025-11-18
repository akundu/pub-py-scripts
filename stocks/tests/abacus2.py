import requests

API_KEY = "s2_7709c4235ed64f2aab529ae65bc0ddde"

# List available models
models_url = "https://apis.abacus.ai/v1/models"
headers = {"Authorization": f"Bearer {API_KEY}"}

response = requests.get(models_url, headers=headers)
print("Available models:")
print(response.json())
