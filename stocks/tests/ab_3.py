import requests
from abacusai import ApiClient
import os

# Get the API key and base URL from the client
client = ApiClient(api_key=os.getenv("ABACUS_API_KEY"))
api_key = client.api_key
base_url = client.api_endpoint

# Try different possible endpoints
endpoints_to_try = [
    '/routellm/chat',
    '/chat',
    '/v0/routellm/chat',
    '/external_application/chat'
]

question = "What are the top 3 tech stocks to watch in 2025?"

for endpoint in endpoints_to_try:
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            headers={
                "apiKey": api_key,
                "Content-Type": "application/json"
            },
            json={
                "message": question,
                "llmName": "claude-sonnet-4"
            }
        )
        
        if response.status_code == 200:
            print(f"✅ Success with endpoint: {endpoint}")
            print(f"Response: {response.json()}")
            break
        else:
            print(f"❌ Failed with {endpoint}: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with {endpoint}: {e}")
