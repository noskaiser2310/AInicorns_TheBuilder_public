"""
Debug script - Kiểm tra chi tiết API request để tìm lỗi 401
"""

import json
import requests
from pathlib import Path

def load_credentials():
    """Load và hiển thị credentials"""
    if not Path("api-keys.json").exists():
        print("❌ File api-keys.json không tồn tại!")
        return None

    with open("api-keys.json", 'r') as f:
        creds = json.load(f)

    # Nếu file chứa một list (một số format export), chuyển về dict theo keys: small, large, embedding
    if isinstance(creds, list):
        normalized = {}
        for entry in creds:
            name = (entry.get('llmApiName') or '').lower()
            if 'small' in name:
                key = 'small'
            elif 'large' in name:
                key = 'large'
            elif 'embed' in name or 'embedding' in name or 'embedings' in name:
                key = 'embedding'
            else:
                # Fallback: try to infer by order (first->small, second->large, third->embedding)
                if 'small' not in normalized:
                    key = 'small'
                elif 'large' not in normalized:
                    key = 'large'
                else:
                    key = 'embedding'

            normalized[key] = {
                'authorization': entry.get('authorization'),
                # support multiple possible key namings
                'token_id': entry.get('tokenId') or entry.get('token_id') or entry.get('tokenID'),
                'token_key': entry.get('tokenKey') or entry.get('token_key') or entry.get('tokenKey'.lower())
            }

        creds = normalized

    return creds

def test_api_detailed(model_type: str, creds: dict):
    """Test API với logging chi tiết"""
    print(f"\n{'='*80}")
    print(f"🔍 TESTING {model_type.upper()} API - CHI TIẾT")
    print('='*80)

    # Get credentials
    cred = creds[model_type]

    # Endpoints mapping
    endpoints = {
        "small": "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small",
        "large": "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large",
        "embedding": "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
    }

    model_names = {
        "small": "vnptai_hackathon_small",
        "large": "vnptai_hackathon_large",
        "embedding": "vnptai_hackathon_embedding"
    }

    endpoint = endpoints[model_type]

    # Print credentials (censored)
    print(f"\n📋 Credentials Check:")
    print(f"   Authorization: {cred['authorization'][:30]}...{cred['authorization'][-20:]}")
    print(f"   Token-ID: {cred['token_id']}")
    print(f"   Token-Key: {cred['token_key'][:30]}...{cred['token_key'][-10:]}")

    # Build headers
    headers = {
        "Authorization": cred["authorization"],
        "Token-id": cred["token_id"],
        "Token-key": cred["token_key"],
        "Content-Type": "application/json"
    }

    print(f"\n📨 Request Details:")
    print(f"   URL: {endpoint}")
    print(f"   Headers: {json.dumps({k: v[:50]+'...' if len(v) > 50 else v for k, v in headers.items()}, indent=6)}")

    # Build payload
    if model_type in ["small", "large"]:
        payload = {
            "model": model_names[model_type],
            "messages": [
                {"role": "user", "content": "Test"}
            ],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 20,
            "n": 1,
            "max_completion_tokens": 10
        }
    else:  # embedding
        payload = {
            "model": model_names[model_type],
            "input": "Test",
            "encoding_format": "float"
        }

    print(f"\n   Payload: {json.dumps(payload, indent=6, ensure_ascii=False)}")

    # Send request
    print(f"\n🚀 Sending request...")

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"\n📥 Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")

        if response.status_code == 200:
            print(f"\n✅ SUCCESS!")
            result = response.json()
            print(f"   Response preview: {str(result)[:200]}...")
            return True
        else:
            print(f"\n❌ FAILED!")
            print(f"   Response Text: {response.text}")

            # Phân tích lỗi
            print(f"\n🔍 Error Analysis:")
            if response.status_code == 401:
                print("   → 401 Unauthorized có thể do:")
                print("      1. Token không đúng (typo khi copy)")
                print("      2. Token hết hạn")
                print("      3. Sai format header (Token-id vs Token-Id)")
                print("      4. Bearer token thiếu 'Bearer ' prefix")
                print("      5. API key chưa được active")

            return False

    except requests.exceptions.RequestException as e:
        print(f"\n❌ REQUEST ERROR: {e}")
        return False

def test_header_variants(model_type: str, creds: dict):
    """Test các variants khác nhau của header names"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING HEADER VARIANTS FOR {model_type.upper()}")
    print('='*80)

    cred = creds[model_type]
    endpoint = f"https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-{model_type}"

    # Các variants có thể
    header_variants = [
        {
            "name": "Standard (theo doc)",
            "headers": {
                "Authorization": cred["authorization"],
                "Token-id": cred["token_id"],
                "Token-key": cred["token_key"],
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Lowercase token-id/key",
            "headers": {
                "Authorization": cred["authorization"],
                "token-id": cred["token_id"],
                "token-key": cred["token_key"],
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Title Case Token-Id/Key",
            "headers": {
                "Authorization": cred["authorization"],
                "Token-Id": cred["token_id"],
                "Token-Key": cred["token_key"],
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Without Bearer prefix",
            "headers": {
                "Authorization": cred["authorization"].replace("Bearer ", ""),
                "Token-id": cred["token_id"],
                "Token-key": cred["token_key"],
                "Content-Type": "application/json"
            }
        }
    ]

    payload = {
        "model": f"vnptai_hackathon_{model_type}",
        "messages": [{"role": "user", "content": "Test"}],
        "temperature": 1.0,
        "max_completion_tokens": 10
    }

    for variant in header_variants:
        print(f"\n🔸 Testing: {variant['name']}")
        try:
            response = requests.post(
                endpoint,
                headers=variant["headers"],
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print(f"   ✅ SUCCESS with variant: {variant['name']}")
                print(f"   → This is the correct header format!")
                return variant["headers"]
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return None

def main():
    print("="*80)
    print("🐛 VNPT API DEBUG - PHÂN TÍCH LỖI 401")
    print("="*80)

    # Load credentials
    print("\n1️⃣ Loading credentials...")
    creds = load_credentials()

    if not creds:
        return

    print("✅ Credentials loaded")

    # Test từng API
    results = {}

    for model_type in ["small", "large", "embedding"]:
        success = test_api_detailed(model_type, creds)
        results[model_type] = success

        # Nếu failed, thử các header variants
        if not success and model_type in ["small", "large"]:
            print(f"\n🔄 Thử các variant header khác...")
            working_headers = test_header_variants(model_type, creds)
            if working_headers:
                results[model_type] = True

    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print('='*80)

    for model, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {model:.<20} {status}")

    if not all(results.values()):
        print(f"\n💡 GỢI Ý FIX:")
        print("   1. Kiểm tra token có hết hạn không (re-download từ portal)")
        print("   2. Verify format header (Token-id vs Token-Id vs token-id)")
        print("   3. Check Bearer prefix trong Authorization")
        print("   4. Liên hệ BTC nếu token mới mà vẫn lỗi")

if __name__ == "__main__":
    main()
