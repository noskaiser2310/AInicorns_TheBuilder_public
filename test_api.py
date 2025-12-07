from vnpt_api_client import VNPTAPIClient


def main():
    log = []
    log.append("=" * 60)
    log.append("QUICK API TEST")
    log.append("=" * 60)
    
    try:
        log.append("\n1. Loading credentials...")
        client = VNPTAPIClient()
        log.append(f"   Keys loaded: {list(client._keys.keys())}")
        
        log.append("\n2. Testing Small LLM...")
        response = client.chat_text([{"role": "user", "content": "Xin chao"}], model="small", max_tokens=30)
        log.append(f"   Response: {response}")

        log.append("\n3. Testing Large LLM...")
        response = client.chat_text([{"role": "user", "content": "Xin chao"}], model="large", max_tokens=30)
        log.append(f"   Response: {response}")

        log.append("\n4. Testing Embedding...")
        emb = client.embed("Viet Nam")
        log.append(f"   Dimension: {len(emb)}")
        
        log.append("\n" + "=" * 60)
        log.append("ALL TESTS PASSED")
        log.append("=" * 60)
        log.append(f"\nQuota used: {client.quota}")
        
        with open("test_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print("Test completed. See test_log.txt")
        return True
        
    except Exception as e:
        log.append(f"\nError: {e}")
        with open("test_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
