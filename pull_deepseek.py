import urllib.request
import json
import sys

def pull_ollama_model(model_name):
    url = 'http://localhost:11434/api/pull'
    data = json.dumps({'name': model_name, 'stream': True}).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                if line:
                    decoded = json.loads(line.decode('utf-8'))
                    print(f"Status: {decoded.get('status', '')}", end='\r')
                    if 'error' in decoded:
                        print(f"\nError: {decoded['error']}")
                        sys.exit(1)
        print(f"\nSuccessfully pulled {model_name}")
    except Exception as e:
        print(f"\nFailed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    pull_ollama_model('deepseek-r1')
