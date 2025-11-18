import requests

# Configuration
API_KEY = "s2_7709c4235ed64f2aab529ae65bc0ddde"
API_URL = "https://apis.abacus.ai/v1/chat/completions"
# Simple test question
question = "What are the top 3 tech stocks to watch in 2025 and why?"

# Prepare the request
payload = {
    "model": "claude-sonnet-4",
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Make the API call
print("Sending question to Abacus API...")
print(f"Question: {question}\n")

response = requests.post(API_URL, headers=headers, json=payload)

# Check response
if response.status_code == 200:
    result = response.json()
    answer = result['choices'][0]['message']['content']
    print("Answer:")
    print(answer)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
