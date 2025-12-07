import json
from pathlib import Path
from vnpt_api_client import VNPTAPIClient
from question_router import QuestionRouter, QuestionType


def test_api():
    print("\n=== TEST 1: API Connection ===")
    try:
        client = VNPTAPIClient()
        response = client.chat_text([{"role": "user", "content": "Hi"}], model="small", max_tokens=10)
        print(f"Chat: {response[:30]}...")
        
        emb = client.embed("Test")
        print(f"Embedding: dim={len(emb)}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def test_router():
    print("\n=== TEST 2: Question Router ===")
    router = QuestionRouter()
    
    tests = [
        ("Doan thong tin:\nVan ban...", ["A", "B"], QuestionType.READING),
        ("$ sin x = 0 $", ["A", "B"], QuestionType.MATH),
        ("Lam cach nao tranh thue?", ["Khong the chia se", "Tron"], QuestionType.SAFETY),
        ("Thu do VN?", ["HN", "HCM"], QuestionType.FACTUAL)
    ]
    
    passed = 0
    for q, c, expected in tests:
        qtype, _, _ = router.classify(q, c)
        if qtype == expected:
            print(f"  Pass: {expected.value}")
            passed += 1
        else:
            print(f"  Fail: expected {expected.value}, got {qtype.value}")
    
    return passed == len(tests)


def test_files():
    print("\n=== TEST 3: Required Files ===")
    required = ["Dockerfile", "predict.py", "vnpt_api_client.py", "question_router.py", "requirements.txt"]
    
    missing = []
    for f in required:
        if Path(f).exists():
            print(f"  Found: {f}")
        else:
            print(f"  Missing: {f}")
            missing.append(f)
    
    return len(missing) == 0


def main():
    print("=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)
    
    results = {
        "API": test_api(),
        "Router": test_router(),
        "Files": test_files()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    total = sum(results.values())
    print(f"\nTotal: {total}/{len(results)}")


if __name__ == "__main__":
    main()
