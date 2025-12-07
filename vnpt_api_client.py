import json
import time
import requests
from typing import List, Dict, Union
from pathlib import Path
from collections import deque
from datetime import datetime


class VNPTAPIClient:
    BASE_URL = "https://api.idg.vnpt.vn"
    
    QUOTA_DAILY = {"small": 1000, "large": 500, "embedding": 500}
    QUOTA_HOURLY = {"small": 60, "large": 40, "embedding": 40}

    def __init__(self, api_keys_file: str = "api-keys.json", cache_dir: str = "./cache"):
        self._keys = {}
        self._load_keys(api_keys_file)
        
        if not self._keys:
            raise ValueError(f"Cannot load API keys from {api_keys_file}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.quota_daily = {"small": 0, "large": 0, "embedding": 0}
        self.call_history = {"small": deque(), "large": deque(), "embedding": deque()}

    def _load_keys(self, filepath: str):
        path = Path(filepath)
        if not path.exists():
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            name = item.get('llmApiName', '').lower()
            auth = item.get('authorization', '')
            
            key_data = {
                'authorization': auth,
                'token_id': item.get('tokenId', ''),
                'token_key': item.get('tokenKey', '')
            }
            
            if 'small' in name:
                self._keys['small'] = key_data
            elif 'large' in name:
                self._keys['large'] = key_data
            elif 'embed' in name:
                self._keys['embedding'] = key_data

    def _headers(self, model: str) -> Dict:
        if model not in self._keys:
            raise ValueError(f"No key for model: {model}")
        k = self._keys[model]
        return {
            "Authorization": k['authorization'],
            "Token-id": k['token_id'],
            "Token-key": k['token_key'],
            "Content-Type": "application/json"
        }

    def _check_rate_limit(self, model: str):
        now = time.time()
        one_hour_ago = now - 3600
        
        history = self.call_history[model]
        while history and history[0] < one_hour_ago:
            history.popleft()
        
        hourly_limit = self.QUOTA_HOURLY.get(model, 60)
        if len(history) >= hourly_limit:
            oldest = history[0]
            wait_time = oldest + 3600 - now + 1
            if wait_time > 0:
                print(f"Rate limit reached for {model}. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
                while history and history[0] < time.time() - 3600:
                    history.popleft()

    def _record_call(self, model: str):
        self.call_history[model].append(time.time())
        self.quota_daily[model] += 1

    def get_quota_status(self) -> Dict:
        now = time.time()
        one_hour_ago = now - 3600
        
        status = {}
        for model in ["small", "large", "embedding"]:
            history = self.call_history[model]
            hourly_used = sum(1 for t in history if t > one_hour_ago)
            status[model] = {
                "daily_used": self.quota_daily[model],
                "daily_limit": self.QUOTA_DAILY[model],
                "hourly_used": hourly_used,
                "hourly_limit": self.QUOTA_HOURLY[model]
            }
        return status

    def chat(self, messages: List[Dict], model: str = "small", 
             temperature: float = 0.7, max_tokens: int = 512, 
             top_p: float = 0.95, top_k: int = 20, seed: int = None) -> Dict:
        
        if model == "small":
            url = f"{self.BASE_URL}/data-service/v1/chat/completions/vnptai-hackathon-small"
            model_name = "vnptai_hackathon_small"
        else:
            url = f"{self.BASE_URL}/data-service/v1/chat/completions/vnptai-hackathon-large"
            model_name = "vnptai_hackathon_large"

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "n": 1
        }
        if seed:
            payload["seed"] = seed

        self._check_rate_limit(model)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = self._headers(model)
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                
                if resp.status_code == 429:
                    wait_time = 60
                    print(f"Rate limit 429, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if resp.status_code == 401 and "rate" in resp.text.lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"Rate limit 401, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                resp.raise_for_status()
                self._record_call(model)
                return resp.json()
                
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"HTTP error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise Exception(f"Max retries exceeded for {model}")

    def chat_text(self, messages: List[Dict], model: str = "small", **kwargs) -> str:
        result = self.chat(messages, model, **kwargs)
        if result.get("choices"):
            return result["choices"][0]["message"]["content"]
        raise ValueError(f"Invalid response: {result}")

    def embed(self, text: Union[str, List[str]]) -> List[float]:
        url = f"{self.BASE_URL}/data-service/vnptai-hackathon-embedding"
        
        self._check_rate_limit("embedding")
        
        single = isinstance(text, str)
        payload = {
            "model": "vnptai_hackathon_embedding",
            "input": text,
            "encoding_format": "float"
        }

        resp = requests.post(url, headers=self._headers("embedding"), json=payload, timeout=30)
        resp.raise_for_status()
        self._record_call("embedding")
        
        result = resp.json()
        if "data" in result and result["data"]:
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings[0] if single else embeddings
        raise ValueError(f"Invalid embedding response: {result}")


if __name__ == "__main__":
    client = VNPTAPIClient()
    print(f"Loaded keys: {list(client._keys.keys())}")
    print(f"Quota status: {client.get_quota_status()}")
