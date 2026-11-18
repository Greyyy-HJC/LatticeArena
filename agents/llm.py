import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get('LLM_BASE', 'https://api.deepseek.com/v1')
THINKING = os.environ.get('LLM_THINKING', 'disabled')
MOCK_QUEUE = []


def _api_key():
    key = os.environ.get('LLM_API_KEY')
    if key:
        return key
    return (Path.home() / 'Desktop' / 'api.key').read_text().strip()


def chat(prompt, model='deepseek-flash'):
    if model == 'mock':
        if not MOCK_QUEUE:
            raise RuntimeError('mock queue is empty')
        return MOCK_QUEUE.pop(0)
    body = json.dumps({'model': model,
                       'messages': [{'role': 'user', 'content': prompt}],
                       'response_format': {'type': 'json_object'},
                       'thinking': {'type': THINKING},
                       'temperature': 0.8}).encode()
    request = urllib.request.Request(BASE + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': f'Bearer {_api_key()}'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())['choices'][0]['message']['content']
