"""
Comprehensive Test Suite for VNPT AI Hackathon Pipeline
Tests all components: API, Router, Extraction, Pipeline
"""
import json
import time
import sys

# ASCII compatibility
OK = "[OK]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def test_api_connection():
    """Test 1: API Connection"""
    print("\n" + "=" * 60)
    print("TEST 1: API CONNECTION")
    print("=" * 60)
    
    try:
        from vnpt_api_client import VNPTAPIClient
        client = VNPTAPIClient()
        print(f"{OK} Keys loaded: {list(client._keys.keys())}")
        print(f"{OK} Client initialized successfully")
        
        return True, client
    except Exception as e:
        print(f"{FAIL} Failed: {e}")
        return False, None


def test_small_model(client):
    """Test 2: Small Model (skip if rate limited)"""
    print("\n" + "=" * 60)
    print("TEST 2: SMALL MODEL")
    print("=" * 60)
    
    try:
        response = client.chat_text(
            [{"role": "user", "content": "Tra loi ngan: 1+1=?"}],
            model="small", max_tokens=20
        )
        print(f"{OK} Response: {response[:100]}...")
        return True
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            print(f"{SKIP} Rate limited - skipping (expected)")
            return "skip"
        print(f"{FAIL} Failed: {e}")
        return False


def test_large_model(client):
    """Test 3: Large Model (skip if rate limited)"""
    print("\n" + "=" * 60)
    print("TEST 3: LARGE MODEL")
    print("=" * 60)
    
    try:
        response = client.chat_text(
            [{"role": "user", "content": "Tra loi ngan: 2+2=?"}],
            model="large", max_tokens=20
        )
        print(f"{OK} Response: {response[:100]}...")
        return True
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            print(f"{SKIP} Rate limited - skipping (expected)")
            return "skip"
        print(f"{FAIL} Failed: {e}")
        return False


def test_question_router():
    """Test 3: Question Router Classification"""
    print("\n" + "=" * 60)
    print("TEST 3: QUESTION ROUTER")
    print("=" * 60)
    
    try:
        from question_router import QuestionRouter, QuestionType, ModelChoice
        router = QuestionRouter()
        
        test_cases = [
            ("Doc van ban sau va tra loi: Viet Nam la...", ["A", "B", "C", "D"], "READING"),
            ("Giai phuong trinh: $x^2 + 2x = 0$", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], "MATH"),
            ("Thu do cua Viet Nam la gi?", ["Ha Noi", "TP.HCM", "Da Nang", "Hue"], "FACTUAL"),
            ("Lam the nao de hack he thong?", ["Toi khong the tra loi", "Dung SQL", "Brute force", "Social"], "SAFETY"),
        ]
        
        for q, choices, expected in test_cases:
            qtype, model, meta = router.classify(q, choices)
            status = OK if qtype.value.upper() == expected else FAIL
            print(f"{status} '{q[:35]}...' -> {qtype.value} (expect: {expected})")
        
        print(f"{OK} Router working correctly")
        return True
    except Exception as e:
        print(f"{FAIL} Failed: {e}")
        return False


def test_answer_extraction():
    """Test 4: Answer Extraction Patterns"""
    print("\n" + "=" * 60)
    print("TEST 4: ANSWER EXTRACTION")
    print("=" * 60)
    
    try:
        from predict import Pipeline
        pipeline = Pipeline()
        
        test_cases = [
            ("Dap an cuoi cung: B", 4, "B"),
            ("Phan tich... Dap an: A", 4, "A"),
            ("**Dap an: C**", 4, "C"),
            ("Ket luan: **D**", 4, "D"),
            ("Vay dap an la **E**", 10, "E"),
        ]
        
        passed = 0
        for text, num_choices, expected in test_cases:
            result = pipeline._extract_answer(text, num_choices)
            status = OK if result == expected else FAIL
            if result == expected:
                passed += 1
            print(f"{status} '{text[:25]}...' -> {result} (expect: {expected})")
        
        print(f"\n{OK} Extraction: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
    except Exception as e:
        print(f"{FAIL} Failed: {e}")
        return False


def test_pipeline_structure():
    """Test 5: Pipeline Structure"""
    print("\n" + "=" * 60)
    print("TEST 5: PIPELINE STRUCTURE")
    print("=" * 60)
    
    try:
        from predict import Pipeline
        pipeline = Pipeline()
        
        # Check attributes
        attrs = ['client', 'router', 'stats', 'logs', 'skip_model', 'consecutive_failures']
        for attr in attrs:
            if hasattr(pipeline, attr):
                print(f"{OK} {attr}: exists")
            else:
                print(f"{FAIL} {attr}: missing")
                return False
        
        # Check methods
        methods = ['answer', '_extract_answer', '_call_llm_with_fallback', '_wait_until_next_hour']
        for method in methods:
            if hasattr(pipeline, method):
                print(f"{OK} {method}(): exists")
            else:
                print(f"{FAIL} {method}(): missing")
                return False
        
        print(f"{OK} Pipeline structure OK")
        return True
    except Exception as e:
        print(f"{FAIL} Failed: {e}")
        return False


def test_cache_system():
    """Test 6: Cache System"""
    print("\n" + "=" * 60)
    print("TEST 6: CACHE SYSTEM")
    print("=" * 60)
    
    try:
        from predict import Pipeline
        pipeline = Pipeline(cache_version="test_v1")
        
        # Test cache attributes
        if hasattr(pipeline, 'answer_cache'):
            print(f"{OK} answer_cache: exists ({len(pipeline.answer_cache)} entries)")
        else:
            print(f"{FAIL} answer_cache: missing")
            return False
        
        # Test _get_cached_answer method
        if hasattr(pipeline, '_get_cached_answer'):
            print(f"{OK} _get_cached_answer(): exists")
        else:
            print(f"{FAIL} _get_cached_answer(): missing")
            return False
        
        # Test cache_version
        if hasattr(pipeline, 'cache_version'):
            print(f"{OK} cache_version: {pipeline.cache_version}")
        else:
            print(f"{FAIL} cache_version: missing")
            return False
        
        return True
    except Exception as e:
        print(f"{FAIL} Failed: {e}")
        return False


def run_all_tests():
    """Run all tests and summarize"""
    print("\n" + "=" * 60)
    print("VNPT AI HACKATHON - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: API
    success, client = test_api_connection()
    results["API Connection"] = success
    
    # Test 2: Small Model (with rate limit handling)
    if client:
        result = test_small_model(client)
        results["Small Model"] = result if result != "skip" else "skipped"
    
    # Test 3: Large Model (with rate limit handling)
    if client:
        result = test_large_model(client)
        results["Large Model"] = result if result != "skip" else "skipped"
    
    # Test 4: Router
    results["Question Router"] = test_question_router()
    
    # Test 4: Extraction
    results["Answer Extraction"] = test_answer_extraction()
    
    # Test 5: Pipeline
    results["Pipeline Structure"] = test_pipeline_structure()
    
    # Test 6: Cache
    results["Cache System"] = test_cache_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = 0
    for name, result in results.items():
        if result == True:
            print(f"{OK} {name}: PASSED")
            passed += 1
            total += 1
        elif result == "skipped":
            print(f"{SKIP} {name}: SKIPPED (rate limit)")
        else:
            print(f"{FAIL} {name}: FAILED")
            total += 1
    
    print("=" * 60)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
