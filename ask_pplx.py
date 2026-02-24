import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

import os

url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ.get('PPLX_API_KEY', 'your_perplexity_api_key_here')}",
    "Content-Type": "application/json"
}
data = {
    "model": "sonar-pro",
    "messages": [
        {"role": "system", "content": "You are a web development research assistant."},
        {"role": "user", "content": "I am building a professional, minimal React-based UI for an AI knowledge base (like NotebookLM). Give me links and source information for the best open-source design systems, component libraries, and React libraries specifically suited for minimal AI chat interfaces and document upload zones. Return exactly 5 highly relevant URLs with brief descriptions."}
    ]
}

req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"))
try:
    with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
        result = json.loads(response.read().decode())
        print(result["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Error: {e}")
