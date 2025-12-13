import json
import time
import requests
from typing import List, Dict, Union
from pathlib import Path
from collections import deque
from datetime import datetime


class VNPTAPIClient:
    BASE_URL = "https://api.idg.vnpt.vn"

    def __init__(self, api_keys_file: str = "api-keys.json", cache_dir: str = "./cache"):
        self._keys = {}
        self._load_keys(api_keys_file)
        
        if not self._keys:
            raise ValueError(f"Cannot load API keys from {api_keys_file}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Track calls for debugging only (no self-imposed limits)
        self.call_count = {"small": 0, "large": 0, "embedding": 0}

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
        # Removed local rate limit check - rely on actual API response for rate limiting
        # The API will return 429 if rate limited, which is handled in chat()
        pass

    def _record_call(self, model: str):
        self.call_count[model] += 1

    def chat(self, messages: List[Dict], model: str = "small", 
             temperature: float = 0, max_tokens: int = 8192, 
             top_p: float = 0.9, top_k: int = 10, seed: int = 42,
             n: int = 1, presence_penalty: float = 0, frequency_penalty: float = 0) -> Dict:
        
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
            "n": n,
            "seed": seed,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }

        self._check_rate_limit(model)

        max_retries = 3  # Increased for timeout retries
        for attempt in range(max_retries):
            try:
                headers = self._headers(model)
                # Timeout: 5 phút (300s) cho cả 2 model - đủ cho câu hỏi phức tạp
                timeout = 300
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if resp.status_code in [429, 401]:
                    raise Exception(f"Rate limit {resp.status_code} for {model}")
                    
                resp.raise_for_status()
                self._record_call(model)
                return resp.json()
                
            except requests.exceptions.Timeout as e:
                # Timeout error - retry with increased wait
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"[TIMEOUT] {model} model timed out, waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                    import time
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Timeout after {max_retries} retries for {model}")
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
        
        raise Exception(f"Max retries exceeded for {model}")

    def _wait_until_next_hour(self) -> float:
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute > 0 or now.second > 0:
            next_hour = next_hour.replace(hour=now.hour + 1)
        wait_seconds = (next_hour - now).total_seconds() + 5
        return max(wait_seconds, 30)

    def chat_text(self, messages: List[Dict], model: str = "small", **kwargs) -> Union[str, List[str]]:
        result = self.chat(messages, model, **kwargs)
        
        # Normal response with choices
        if result.get("choices"):
            if len(result["choices"]) == 1:
                return result["choices"][0]["message"]["content"]
            else:
                return [c["message"]["content"] for c in result["choices"]]
        
        # Check for encoded response (VNPT format with dataBase64)
        if result.get("dataBase64"):
            import base64
            try:
                decoded = base64.b64decode(result["dataBase64"]).decode('utf-8')
                import json
                data = json.loads(decoded)
                
                # Check for error in decoded data
                if data.get("error"):
                    error_code = data["error"].get("code", 0)
                    error_msg = data["error"].get("message", "Unknown error")
                    
                    # Content filter (400) - return a safe response indicator
                    if error_code == 400:
                        raise ValueError(f"Content filtered: {error_msg[:100]}")
                    else:
                        raise ValueError(f"API error {error_code}: {error_msg[:100]}")
                
                # Check for choices in decoded data
                if data.get("choices"):
                    if len(data["choices"]) == 1:
                        return data["choices"][0]["message"]["content"]
                    else:
                        return [c["message"]["content"] for c in data["choices"]]
                        
            except Exception as e:
                if "Content filtered" in str(e) or "API error" in str(e):
                    raise
                print(f"[DEBUG] Failed to decode dataBase64: {e}")
        
        # Fallback: unknown response format
        print(f"[DEBUG] API response has no choices. Keys: {result.keys()}")
        print(f"[DEBUG] Response preview: {str(result)[:300]}")
        raise ValueError(f"Invalid response: {str(result)[:200]}")

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
