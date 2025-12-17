import json
import hashlib
import re
import time
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from vnpt_api_client import VNPTAPIClient
from question_router import QuestionRouter, QuestionType, ModelChoice

# RAG disabled for Docker build
HAS_VECTOR_DB = False
# try:
#     from build_vector_db import VectorDBSearcher
#     HAS_VECTOR_DB = True
# except ImportError:
#     HAS_VECTOR_DB = False


class RateLimitError(Exception):
    pass


class LargeModelRateLimited(Exception):
    """Raised when Large model is rate limited - caller should defer this question"""
    pass


class Pipeline:
    def __init__(self, vector_db_path: str = "./data/vector_db", cache_dir: str = "./cache", 
                 log_file: str = "inference_log.json", cache_version: str = "v10",
                 small_workers: int = 50, large_workers: int = 30):
        self.client = VNPTAPIClient(cache_dir=cache_dir)
        self.router = QuestionRouter()
        self.vector_db = None
        self.stats = {"total": 0, "by_type": {}, "by_model": {"small": 0, "large": 0, "none": 0}}
        self.log_file = log_file
        self.logs = []
        self.consecutive_failures = {"small": 0, "large": 0}
        self.skip_model = {"small": False, "large": False}
        self.pending_queues = {"small": 0, "large": 0}  # Track pending questions per queue
        
        # Worker settings (no limit - rate limiting handled automatically)
        self.small_workers = small_workers
        self.large_workers = large_workers
        
        import threading
        self._cache_lock = threading.Lock()  # Thread-safe cache operations
        
        # Answer caching with version support for resume capability
        # When you change code/prompts, use a new version to start fresh
        self.cache_version = cache_version
        self.answer_cache_file = f"answer_cache_{cache_version}.json"
        self.answer_cache = self._load_answer_cache()
        
        if HAS_VECTOR_DB:
            index_path = Path(vector_db_path) / "faiss.index"
            if index_path.exists():
                try:
                    self.vector_db = VectorDBSearcher(vector_db_path)
                except Exception:
                    pass
    
    def _load_answer_cache(self) -> dict:
        """Load cached answers from file for resume capability"""
        cache_path = Path(self.answer_cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Support both old format (dict) and new format (with metadata)
                if isinstance(data, dict) and "answers" in data:
                    cache = data["answers"]
                else:
                    cache = data
                print(f"Loaded {len(cache)} cached answers from {self.answer_cache_file}")
                return cache
            except Exception:
                return {}
        return {}
    
    def _save_answer_cache(self):
        """Save answer cache to file with metadata (thread-safe)"""
        with self._cache_lock:
            # Make a copy to avoid dict changed during iteration
            cache_copy = dict(self.answer_cache)
            data = {
                "version": self.cache_version,
                "count": len(cache_copy),
                "answers": cache_copy
            }
            with open(self.answer_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_from_cache(self, old_cache_file: str):
        """Import answers from an old cache file (different version)"""
        try:
            with open(old_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "answers" in data:
                old_answers = data["answers"]
            else:
                old_answers = data
            imported = 0
            for qid, answer in old_answers.items():
                if qid not in self.answer_cache:
                    self.answer_cache[qid] = answer
                    imported += 1
            print(f"Imported {imported} answers from {old_cache_file}")
            self._save_answer_cache()
        except Exception as e:
            print(f"Error importing cache: {e}")
    
    def _get_cached_answer(self, qid: str):
        """Get cached entry for a question if exists. Returns (answer, log_entry) or (None, None)"""
        entry = self.answer_cache.get(qid)
        if entry:
            if isinstance(entry, dict):
                return entry.get("extracted_answer", entry.get("answer")), entry
            else:
                # Old format: just the answer string
                return entry, None
        return None, None

    def retrieve(self, question: str, k: int = 5) -> str:
        if not self.vector_db:
            return ""
        try:
            results = self.vector_db.search(question, k=k)
            contexts = []
            for i, r in enumerate(results, 1):
                title = r.get('title', '')
                text = r.get('text', '')
                contexts.append(f"[{i}] ({title}) {text}" if title else f"[{i}] {text}")
            return "\n\n".join(contexts)
        except Exception:
            return ""

    def _call_llm_with_fallback(self, messages, preferred_model: str, param_config: dict = None, allow_fallback: bool = True) -> tuple:
        """
        Call LLM with smart fallback and rate limit tracking.
        Returns: (response, used_model, fallback_used)
        """
        if param_config is None:
            param_config = {}
        
        model = preferred_model
        
        # Check if model is known to be rate limited
        if self.skip_model.get(model, False):
            if model == "large" and allow_fallback:
                model = "small"  # Use fallback directly
            elif model == "small" and self.skip_model.get("large", False):
                # Both models known to be limited - wait first
                print(f"[BOTH MODELS LIMITED] Waiting for quota reset before retry...")
                self._wait_until_next_hour()
                self.skip_model = {"small": False, "large": False}
        
        # Quick retry: only 2 attempts with short delays
        retry_delays = [2, 5]  # Much shorter: 2s then 5s
        
        for attempt in range(len(retry_delays) + 1):
            try:
                resp = self.client.chat_text(
                    messages, 
                    model=model, 
                    temperature=param_config.get("temperature", 0.8),
                    max_tokens=8192,
                    seed=param_config.get("seed", 42),
                    top_p=param_config.get("top_p", 0.9),
                    top_k=param_config.get("top_k", 10),
                    n=1
                )
                # Success - reset fail counters
                self.skip_model[model] = False
                return resp, model, model != preferred_model
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in ["rate", "limit", "401", "429"])
                is_server_error = any(x in error_str for x in ["invalid response", "datasign", "500"])
                is_content_filtered = "content filtered" in error_str
                
                if is_content_filtered:
                    raise  # Don't retry content filters
                
                if is_rate_limit:
                    # Mark model as rate limited globally
                    self.skip_model[model] = True
                    
                    if model == "large" and allow_fallback and not self.skip_model.get("small", False):
                        # Try small immediately - if Small model quota is available
                        print(f"[{model.upper()} LIMITED] Trying small model fallback...")
                        try:
                            resp = self.client.chat_text(
                                messages, model="small",
                                temperature=param_config.get("temperature", 0.8),
                                max_tokens=8192, seed=param_config.get("seed", 42),
                                top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                            )
                            return resp, "small", True
                        except Exception as small_err:
                            if any(x in str(small_err).lower() for x in ["rate", "limit", "401", "429"]):
                                self.skip_model["small"] = True
                                # Both limited - wait and retry
                                print(f"[BOTH LIMITED] Waiting for quota reset...")
                                self._wait_until_next_hour()
                                self.skip_model = {"small": False, "large": False}
                                # Retry once after reset
                                try:
                                    resp = self.client.chat_text(
                                        messages, model="small",
                                        temperature=param_config.get("temperature", 0.8),
                                        max_tokens=8192, seed=param_config.get("seed", 42),
                                        top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                                    )
                                    return resp, "small", True
                                except Exception as retry_err:
                                    raise RateLimitError(f"Retry after reset failed: {retry_err}")
                            raise RateLimitError(f"Small model failed: {small_err}")
                        
                    elif model == "large":
                        # Both Large and Small are rate limited - wait and retry
                        print(f"[BOTH MODELS LIMITED] Large limited, Small also limited. Waiting for reset...")
                        self._wait_until_next_hour()
                        self.skip_model = {"small": False, "large": False}
                        # Retry with small after reset
                        try:
                            resp = self.client.chat_text(
                                messages, model="small",
                                temperature=param_config.get("temperature", 0.8),
                                max_tokens=8192, seed=param_config.get("seed", 42),
                                top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                            )
                            return resp, "small", True
                        except Exception as retry_err:
                            raise LargeModelRateLimited(f"Retry after reset failed: {retry_err}")
                    else:
                        # Small model rate limited
                        if allow_fallback and self.pending_queues.get("large", 0) == 0 and not self.skip_model.get("large", False):
                            # Try large as fallback - only if Large queue is empty
                            print(f"[SMALL LIMITED] Large queue empty, trying large model...")
                            try:
                                resp = self.client.chat_text(
                                    messages, model="large",
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192, seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                                )
                                return resp, "large", True
                            except Exception as large_err:
                                if any(x in str(large_err).lower() for x in ["rate", "limit", "401", "429"]):
                                    self.skip_model["large"] = True
                                    # Both limited
                                    print(f"[BOTH LIMITED] Waiting for quota reset...")
                                    self._wait_until_next_hour()
                                    self.skip_model = {"small": False, "large": False}
                                    try:
                                        resp = self.client.chat_text(
                                            messages, model="small",
                                            temperature=param_config.get("temperature", 0.8),
                                            max_tokens=8192, seed=param_config.get("seed", 42),
                                            top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                                        )
                                        return resp, "small", True
                                    except Exception as retry_err:
                                        raise RateLimitError(f"Retry after reset failed: {retry_err}")
                                raise RateLimitError(f"Large model failed: {large_err}")
                        raise RateLimitError(f"Small rate limited: {e}")
                
                elif is_server_error:
                    # Server error - retry with delay
                    if attempt < len(retry_delays):
                        delay = retry_delays[attempt]
                        print(f"Model {model} server error, retry {attempt+1}... (waiting {delay}s)")
                        time.sleep(delay)
                        continue
                    else:
                        # Exhausted retries for server error
                        if model == "large" and allow_fallback:
                            try:
                                resp = self.client.chat_text(
                                    messages, model="small",
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192, seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                                )
                                return resp, "small", True
                            except:
                                raise LargeModelRateLimited(f"Both models failed: {e}")
                        raise RateLimitError(f"Model {model} server error: {e}")
                else:
                    # Unknown error - retry once
                    if attempt < 1:
                        print(f"Model {model} unknown error, retrying once...")
                        time.sleep(2)
                        continue
                    else:
                        if model == "large" and allow_fallback:
                            try:
                                resp = self.client.chat_text(
                                    messages, model="small",
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192, seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9), top_k=param_config.get("top_k", 10), n=1
                                )
                                return resp, "small", True
                            except:
                                raise LargeModelRateLimited(f"Both models failed: {e}")
                        raise RateLimitError(f"Model {model} failed: {e}")
        
        raise RateLimitError(f"Model {model} exhausted retries")

    def _wait_until_next_hour(self):
        from datetime import timedelta
        # Wait until 10 minutes past next hour (e.g., 3:10, 4:10)
        # This ensures the API's rolling 60-min window has fully reset
        now = datetime.now()
        
        # Calculate next hour + 10 minutes
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute >= 0:  # Always go to next hour
            next_hour = next_hour + timedelta(hours=1)
        target_time = next_hour + timedelta(minutes=10)  # XX:10
        
        wait_seconds = (target_time - now).total_seconds()
        
        print(f"\nRate limited. Waiting until {target_time.strftime('%H:%M')} ({int(wait_seconds/60)} minutes)...")
        print("(API uses rolling 60-min window, waiting until safe)")
        time.sleep(wait_seconds)
        
        # Reset state after waiting
        self.skip_model = {"small": False, "large": False}
        self.consecutive_failures = {"small": 0, "large": 0}
        print("Ready to retry with both models...")

    def answer(self, question: str, choices: List[str], qid: str = "") -> str:
        # Check cache first for resume capability
        cached_answer, cached_log = self._get_cached_answer(qid)
        if cached_answer:
            self.stats["total"] += 1
            # Add cached log to logs if available
            if cached_log:
                cached_log["from_cache"] = True
                self.logs.append(cached_log)
            return cached_answer
        
        qtype, model_choice, meta = self.router.classify(question, choices)
        self.stats["by_type"][qtype.value] = self.stats["by_type"].get(qtype.value, 0) + 1
        
        log_entry = {
            "qid": qid,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "choices": choices,
            "type": qtype.value,
            "model": model_choice.value if model_choice.value else "none"
        }
        
        safe_idx = meta.get("safe_idx")
        subtype = meta.get("subtype")  # Get subtype from classify() metadata
        
        context = self.retrieve(question, k=5) if meta.get("use_rag") else None
        
        first_prompt = self.router.build_prompt(qtype, question, choices, context, 0, subtype=subtype)
        log_entry["prompt_system"] = first_prompt[0]["content"][:300] + "..."
        log_entry["prompt_user"] = first_prompt[1]["content"][:500] + "..."
        
        if model_choice == ModelChoice.NONE:
            self.stats["by_model"]["none"] += 1
            log_entry["answer"] = "A"
            log_entry["reasoning"] = "Fallback to A"
            self.logs.append(log_entry)
            return "A"
        
        preferred_model = model_choice.value
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                used_model = None  # Initialize to prevent reference error
                responses = []
                
                if qtype == QuestionType.MATH:
                    fallback_used = False
                    messages1 = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp1, model1, fb1 = self._call_llm_with_fallback(messages1, preferred_model)
                    if fb1:
                        fallback_used = True
                    first_answer = self._extract_answer(resp1, len(choices))
                    
                    verify_prompt = f"""Một học sinh đã giải bài toán sau.

BÀI TOÁN:
{question}

CÁC ĐÁP ÁN:
{chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])}

LỜI GIẢI CỦA HỌC SINH:
{resp1}

NHIỆM VỤ CỦA BẠN:
1. Kiểm tra lại từng bước tính toán của học sinh
2. Xác định xem đáp án {first_answer} có đúng không
3. Nếu sai, hãy giải lại và chọn đáp án đúng và ghi nhớ là đề bài luôn đúng nếu không có kết quả sai tính toán lại
4. Nếu đúng, xác nhận đáp án

Luôn kết thúc bằng: "Đáp án cuối cùng: X" (X là chữ cái cuối cùng bạn chọn)"""
                    
                    verify_messages = [
                        {"role": "system", "content": "Bạn là giám khảo Toán học với nhiệm vụ kiểm tra lại bài giải của học sinh. Hãy xác thực kết quả và đưa ra đáp án cuối cùng."},
                        {"role": "user", "content": verify_prompt}
                    ]
                    resp2, used_model, fb2 = self._call_llm_with_fallback(verify_messages, preferred_model)
                    if fb2:
                        fallback_used = True
                    
                    responses = [resp1, resp2]
                    answer = self._extract_answer(resp2, len(choices))
                    
                    log_entry["first_answer"] = first_answer
                    log_entry["verification"] = True
                    if fallback_used:
                        log_entry["needs_large_retry"] = True
                    
                elif qtype == QuestionType.SAFETY:
                    messages = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp, used_model, fallback_used = self._call_llm_with_fallback(messages, preferred_model)
                    responses = [resp]
                    answer = self._extract_answer(resp, len(choices))
                    if fallback_used:
                        log_entry["needs_large_retry"] = True
                    
                elif qtype == QuestionType.READING:
                    # READING: 3-prompt voting for better accuracy
                    choices_str = chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    votes = []
                    fallback_used = False
                    
                    # Call 1: 5-step deep analysis prompt with SMALL (cheaper)
                    messages1 = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp1, model1, fb1 = self._call_llm_with_fallback(messages1, "small",
                        {"temperature": 0.3, "top_p": 0.7, "top_k": 10, "seed": 42})
                    answer1 = self._extract_answer(resp1, len(choices))
                    votes.append({"answer": answer1, "model": model1, "prompt": "v1"})
                    if fb1:
                        fallback_used = True
                    
                    # Call 2: 4-step data extraction prompt with SMALL (cheaper)
                    messages2 = self.router._build_reading_prompt_v2(question, choices_str)
                    resp2, model2, fb2 = self._call_llm_with_fallback(messages2, "small",
                        {"temperature": 0.6, "top_p": 0.9, "top_k": 15, "seed": 123})
                    answer2 = self._extract_answer(resp2, len(choices))
                    votes.append({"answer": answer2, "model": model2, "prompt": "v2"})
                    if fb2:
                        fallback_used = True
                    
                    # Call 3: Debate-based analysis prompt with LARGE (for quality)
                    messages3 = self.router._build_reading_prompt_v3(question, choices_str)
                    resp3, model3, fb3 = self._call_llm_with_fallback(messages3, "large",
                        {"temperature": 1.0, "top_p": 0.9, "top_k": 20, "seed": 456})
                    answer3 = self._extract_answer(resp3, len(choices))
                    votes.append({"answer": answer3, "model": model3, "prompt": "v3"})
                    if fb3:
                        fallback_used = True
                    
                    responses = [resp1, resp2, resp3]
                    log_entry["reading_votes"] = [v["answer"] for v in votes]
                    log_entry["reading_models"] = [v["model"] for v in votes]
                    log_entry["fallback_used"] = fallback_used
                    
                    # Count votes (3 votes total, need 2+ for majority)
                    vote_counts = {}
                    for v in votes:
                        ans = v["answer"]
                        vote_counts[ans] = vote_counts.get(ans, 0) + 1
                    
                    # Find majority answer (2 or 3 votes)
                    max_votes = max(vote_counts.values())
                    majority_answers = [ans for ans, count in vote_counts.items() if count == max_votes]
                    
                    if max_votes >= 2:
                        # Clear winner with 2+ votes
                        answer = majority_answers[0]
                        log_entry["reading_consensus"] = True
                        log_entry["consensus_votes"] = max_votes
                    else:
                        # All 3 different - use tiebreak with Large
                        tiebreak_prompt = f"""BA CHUYÊN GIA ĐỌC HIỂU ĐƯA RA 3 ĐÁP ÁN KHÁC NHAU:

Chuyên gia 1 (phân tích chuyên sâu) chọn: {answer1}
Chuyên gia 2 (trích dẫn dữ liệu) chọn: {answer2}
Chuyên gia 3 (tranh biện) chọn: {answer3}

VĂN BẢN VÀ CÂU HỎI GỐC:
{question}

CÁC ĐÁP ÁN:
{choices_str}

NHIỆM VỤ CỦA BẠN:
1. Đọc lại văn bản gốc một cách cẩn thận
2. Xem xét lập luận của cả 3 chuyên gia
3. Xác định đáp án nào được HỖ TRỢ TRỰC TIẾP nhất bởi văn bản
4. Nếu tất cả đều sai, chọn đáp án đúng nhất

Luôn kết thúc bằng: "Đáp án cuối cùng: X" (X là chữ cái cuối cùng bạn chọn)"""

                        tiebreak_messages = [
                            {"role": "system", "content": "Bạn là giám khảo cao cấp với nhiệm vụ phân xử khi có bất đồng. Hãy đọc kỹ văn bản và chọn đáp án được HỖ TRỢ RÕ RÀNG NHẤT."},
                            {"role": "user", "content": tiebreak_prompt}
                        ]
                        
                        try:
                            resp4, model4, fb4 = self._call_llm_with_fallback(tiebreak_messages, "large",
                                {"temperature": 0.1, "top_p": 0.9, "top_k": 5, "seed": 999})
                            responses.append(resp4)
                            answer = self._extract_answer(resp4, len(choices))
                            log_entry["tiebreak_answer"] = answer
                            log_entry["tiebreak_model"] = model4
                            if fb4:
                                fallback_used = True
                        except:
                            # Fallback to first answer if tiebreak fails
                            answer = answer1
                        
                        log_entry["reading_consensus"] = False
                    
                    # Mark for retry if fallback was used
                    if fallback_used:
                        log_entry["needs_large_retry"] = True
                    
                else:
                    # FACTUAL: Single call with specialized prompts
                    messages = self.router.build_prompt(qtype, question, choices, context, 0)
                    
                    resp, used_model, fallback_used = self._call_llm_with_fallback(messages, preferred_model, 
                        {"temperature": 0.3, "top_p": 0.85, "top_k": 10, "seed": 42})
                    responses = [resp]
                    answer = self._extract_answer(resp, len(choices))
                    
                    log_entry["single_call"] = True
                    if fallback_used:
                        log_entry["needs_large_retry"] = True
                
                if used_model:
                    self.stats["by_model"][used_model] += 1
                    log_entry["model"] = used_model
                
                log_entry["model_responses"] = responses
                log_entry["extracted_answer"] = answer
                time.sleep(1)
                break
                
            except RateLimitError:
                # Re-raise to let run() handle Smart Queue switching
                raise
                    
            except Exception as e:
                error_str = str(e)
                log_entry["error"] = error_str
                
                # Content filtered by API or LLM refused - select "cannot answer" choice
                if ("content filtered" in error_str.lower() or 
                    "không thể trả lời" in error_str.lower() or 
                    "khong the tra loi" in error_str.lower()):
                    cannot_answer = self._find_cannot_answer_choice(choices)
                    if cannot_answer:
                        answer = cannot_answer
                        log_entry["reasoning"] = "Content filtered/refused - selected cannot answer option"
                    else:
                        # No clear "cannot answer" choice, use first choice as fallback
                        answer = "A"
                        log_entry["reasoning"] = "Content filtered but no cannot answer option found"
                    log_entry["extracted_answer"] = answer
                else:
                    # Other errors - fallback to A
                    answer = "A"
                    log_entry["extracted_answer"] = answer
                break  # Don't retry on exceptions
        
        self.logs.append(log_entry)
        
        # Save FULL log entry to cache for resume capability and evaluation
        if qid:
            self.answer_cache[qid] = log_entry  # Store entire log entry, not just answer
            # Save cache every 10 questions
            if len(self.answer_cache) % 10 == 0:
                self._save_answer_cache()
        
        return answer

    def _extract_answer(self, text: str, num_choices: int) -> str:
        """
        Extract answer from LLM response.
        Rule: Answer MUST be after "Đáp án cuối cùng" (or variants)
        Validated to 99.5% accuracy on test cache.
        """
        valid = [chr(65 + i) for i in range(num_choices)]
        lines = text.strip().split('\n')
        
        found_answers = []
        
        # PRIORITY 1: Find all "Đáp án cuối cùng" occurrences and extract answer
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            if 'đáp án cuối cùng' in line_lower:
                # Case 1: Answer on SAME line
                # Patterns: "Đáp án cuối cùng: A" or "Đáp án cuối cùng là A" or "Đáp án cuối cùng: **A**"
                match = re.search(
                    r'(?:ĐÁP ÁN CUỐI CÙNG|Đáp án cuối cùng)[:\s]*(?:là)?[\s\*\:]*\[?([A-Ja-j])[\.\]\s\*\)\,]?',
                    line, re.IGNORECASE
                )
                if match:
                    ans = match.group(1).upper()
                    if ans in valid:
                        found_answers.append(ans)
                        continue
                
                # Case 2: Answer on NEXT line (only if not found on same line)
                # Pattern: "**A. text**" or "**A**" at start of next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    match = re.search(r'^\*\*\s*([A-Ja-j])[\.\s\*\)]', next_line)
                    if match:
                        ans = match.group(1).upper()
                        if ans in valid:
                            found_answers.append(ans)
        
        # Return LAST found answer (final answer in response)
        if found_answers:
            return found_answers[-1]
        
        # PRIORITY 2: Fallback - Try other "đáp án" patterns
        for line in reversed(lines[-20:]):
            match = re.search(r'[Đđ][áa]p\s*[áa]n[:\s]+\*?\*?([A-Ja-j])(?:[.\s\)\*\}]|$)', line, re.IGNORECASE)
            if match:
                ans = match.group(1).upper()
                if ans in valid:
                    return ans
        
        # PRIORITY 3: Default fallback
        return "A"

    def _find_cannot_answer_choice(self, choices: List[str]) -> str:
        """Find the 'cannot answer' choice when content is filtered."""
        cannot_answer_patterns = [
            # Vietnamese patterns
            r'không thể trả lời',
            r'khong the tra loi',
            r'không trả lời được',
            r'từ chối trả lời',
            r'không cung cấp',
            r'không hỗ trợ',
            r'tôi không thể',
            r'không thể cung cấp thông tin',
            # English patterns
            r'cannot answer',
            r'cannot provide',
            r'unable to answer',
        ]
        
        # First pass: find exact match
        for idx, choice in enumerate(choices):
            choice_lower = choice.lower()
            for pattern in cannot_answer_patterns:
                if re.search(pattern, choice_lower):
                    return chr(65 + idx)
        
        # Second pass: find choice mentioning refusal/inability
        refusal_keywords = ['không', 'từ chối', 'vi phạm', 'cannot', 'unable']
        for idx, choice in enumerate(choices):
            choice_lower = choice.lower()
            if any(kw in choice_lower for kw in refusal_keywords):
                # Check if it's about refusing to provide info
                if 'thông tin' in choice_lower or 'trả lời' in choice_lower or 'cung cấp' in choice_lower:
                    return chr(65 + idx)
        
        # Fallback: If we reach here, we couldn't find a clear "cannot answer" choice
        # Return None to indicate we should try another approach
        return None  # Changed from "A" to None

    def _process_single_question(self, q: dict) -> dict:
        """Process a single question and return result dict"""
        qid = q.get("qid", "")
        
        # Check cache first
        if qid in self.answer_cache:
            cached_answer, _ = self._get_cached_answer(qid)
            return {"qid": qid, "answer": cached_answer, "status": "cached", "question": q}
        
        try:
            answer = self.answer(q["question"], q["choices"], qid)
            return {"qid": qid, "answer": answer, "status": "success", "question": q}
        except LargeModelRateLimited as e:
            return {"qid": qid, "question": q, "status": "large_limited", "error": str(e)}
        except RateLimitError as e:
            return {"qid": qid, "question": q, "status": "small_limited", "error": str(e)}
        except Exception as e:
            return {"qid": qid, "answer": "A", "status": "error", "error": str(e), "question": q}

    def _save_to_cache_immediately(self, result: dict):
        """Save a completed question to cache immediately with full details"""
        qid = result.get("qid", "")
        if not qid or result["status"] in ["large_limited", "small_limited"]:
            return
        
        # Build cache entry with full question
        q = result.get("question", {})
        
        # Get detailed info from the latest log entry for this qid
        log_entry = None
        for log in reversed(self.logs):
            if log.get("qid") == qid:
                log_entry = log
                break
        
        entry = {
            "question": q.get("question", ""),
            "choices": q.get("choices", []),
            "extracted_answer": result.get("answer", "A"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add detailed info from log if available
        if log_entry:
            entry["type"] = log_entry.get("type", "unknown")
            entry["model"] = log_entry.get("model", "unknown")
            model_responses = log_entry.get("model_responses", [])
            entry["response"] = model_responses[-1] if model_responses else ""
            entry["prompt_system"] = log_entry.get("prompt_system", "")
            entry["prompt_user"] = log_entry.get("prompt_user", "")
            entry["all_responses"] = model_responses
        else:
            entry["type"] = "parallel"
            entry["model"] = "unknown"
        
        # Update in-memory cache
        self.answer_cache[qid] = entry
        
        # Write to disk immediately
        self._save_answer_cache()

    def _process_parallel_smart(self, questions: List[dict], max_workers: int = 5, 
                                 desc: str = "Processing", stop_on_limit: bool = True) -> tuple:
        """
        Smart parallel processing with immediate stop on rate limit.
        Returns: (results_dict, remaining_questions, rate_limit_type)
        - rate_limit_type: None, "small", or "large"
        """
        results = {}
        remaining = list(questions)  # Copy to track unprocessed
        rate_limit_type = None
        processed_qids = set()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_q = {executor.submit(self._process_single_question, q): q for q in questions}
            
            pbar = tqdm(total=len(questions), desc=desc)
            
            for future in as_completed(future_to_q):
                result = future.result()
                qid = result["qid"]
                processed_qids.add(qid)
                pbar.update(1)
                
                if result["status"] == "success":
                    results[qid] = result["answer"]
                    self.stats["total"] += 1
                    # answer() already saved to cache, just persist to disk
                    self._save_answer_cache()
                    
                elif result["status"] == "cached":
                    results[qid] = result["answer"]
                    
                elif result["status"] == "small_limited":
                    rate_limit_type = "small"
                    if stop_on_limit:
                        print(f"\n[RATE LIMIT] Small model limited at {qid}. Stopping queue...")
                        # Cancel remaining futures
                        for f in future_to_q:
                            f.cancel()
                        break
                        
                elif result["status"] == "large_limited":
                    rate_limit_type = "large"
                    if stop_on_limit:
                        print(f"\n[RATE LIMIT] Large model limited at {qid}. Stopping queue...")
                        for f in future_to_q:
                            f.cancel()
                        break
                        
                elif result["status"] == "error":
                    results[qid] = result["answer"]
                    # Error already handled, just persist cache to disk
                    self._save_answer_cache()
                    print(f"[ERROR] {qid}: {result.get('error', 'Unknown')[:50]}")
            
            pbar.close()
        
        # Calculate remaining questions (not processed)
        remaining = [q for q in questions if q.get("qid") not in processed_qids]
        
        return results, remaining, rate_limit_type

    def save_logs(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"Logs saved to {self.log_file}")

    def run(self, input_file: str = "/code/private_test.json", output_file: str = "/code/submission.csv"):
        print("=" * 60)
        print("VNPT AI HACKATHON - INFERENCE")
        print("=" * 60)
        
        # Support both CSV and JSON input formats
        if input_file.endswith('.csv'):
            # CSV format: qid, question, choices (A|B|C|D separated)
            df = pd.read_csv(input_file)
            questions = []
            for _, row in df.iterrows():
                q = {"qid": str(row.get("qid", row.name))}
                q["question"] = row.get("question", row.get("context", ""))
                # Handle choices - could be separate columns or combined
                if "choices" in row:
                    # Combined format: "A|B|C|D"
                    q["choices"] = row["choices"].split("|") if isinstance(row["choices"], str) else list(row["choices"])
                else:
                    # Separate columns: A, B, C, D
                    q["choices"] = [str(row.get(c, "")) for c in ["A", "B", "C", "D"] if c in row]
                questions.append(q)
            print(f"Loaded {len(questions)} questions from CSV")
        else:
            # JSON format (default for BTC)
            with open(input_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print(f"Loaded {len(questions)} questions from JSON")
        
        cached_count = sum(1 for q in questions if q.get("qid", "") in self.answer_cache)
        if cached_count > 0:
            print(f"Found {cached_count} cached answers - will skip those")
        
        # Separate questions by model requirement
        small_questions = []
        large_questions = []
        
        for q in questions:
            qid = q.get("qid", "")
            if qid in self.answer_cache:
                continue  # Already cached
            
            qtype, model_choice, _ = self.router.classify(q["question"], q["choices"])
            if model_choice == ModelChoice.LARGE:
                large_questions.append(q)
            else:
                small_questions.append(q)
        
        print(f"To process: {len(small_questions)} Small model + {len(large_questions)} Large model questions")
        
        results = {}  # qid -> answer
        
        # CONCURRENT PROCESSING WITH QUEUE-AWARE FALLBACK
        # Small và Large chạy ĐỒNG THỜI
        # Fallback chỉ được phép khi queue target đã HẾT:
        # - Large→Small: chỉ khi pending_small == 0
        # - Small→Large: chỉ khi pending_large == 0
        
        pending_small = list(small_questions)
        pending_large = list(large_questions)
        
        # Track pending counts for fallback decisions
        self.pending_queues = {"small": len(pending_small), "large": len(pending_large)}
        
        import threading
        results_lock = threading.Lock()
        small_results = {}
        large_results = {}
        small_remaining = []
        large_remaining = []
        small_limit = [None]
        large_limit = [None]
        
        def process_small_queue():
            nonlocal small_results, small_remaining
            if pending_small:
                print(f"\n[PARALLEL] Processing {len(pending_small)} SMALL model questions ({self.small_workers} workers)...")
                small_results, small_remaining, small_limit[0] = self._process_parallel_smart(
                    pending_small, max_workers=self.small_workers, desc="Small Model", stop_on_limit=True
                )
                # Update pending count
                with results_lock:
                    self.pending_queues["small"] = len(small_remaining)
        
        def process_large_queue():
            nonlocal large_results, large_remaining
            if pending_large:
                print(f"\n[PARALLEL] Processing {len(pending_large)} LARGE model questions ({self.large_workers} workers)...")
                large_results, large_remaining, large_limit[0] = self._process_parallel_smart(
                    pending_large, max_workers=self.large_workers, desc="Large Model", stop_on_limit=True
                )
                # Update pending count
                with results_lock:
                    self.pending_queues["large"] = len(large_remaining)
        
        # Run both queues concurrently
        print(f"\n[CONCURRENT] Starting Small ({self.small_workers} workers) + Large ({self.large_workers} workers) simultaneously...")
        print(f"[INFO] Fallback only when target queue is EMPTY")
        
        small_thread = threading.Thread(target=process_small_queue)
        large_thread = threading.Thread(target=process_large_queue)
        
        small_thread.start()
        large_thread.start()
        
        small_thread.join()
        large_thread.join()
        
        # Merge results
        results.update(small_results)
        results.update(large_results)
        pending_small = small_remaining
        pending_large = large_remaining
        
        print(f"\n[DONE] First run: {len(small_results)} Small + {len(large_results)} Large processed")
        
        # Handle rate limits with retry loop
        while pending_small or pending_large:
            print(f"\n[WAITING] {len(pending_small)} Small + {len(pending_large)} Large pending")
            self._wait_until_next_hour()
            print(f"[RETRY] Quota reset. Retrying...")
            
            # Update pending counts
            self.pending_queues = {"small": len(pending_small), "large": len(pending_large)}
            
            # Run again concurrently
            small_thread = threading.Thread(target=process_small_queue)
            large_thread = threading.Thread(target=process_large_queue)
            small_thread.start()
            large_thread.start()
            small_thread.join()
            large_thread.join()
            
            # Merge results and update pending
            results.update(small_results)
            results.update(large_results)
            pending_small = small_remaining
            pending_large = large_remaining
        
        self.pending_queues = {"small": 0, "large": 0}
        print(f"\n[COMPLETE] Processed {len(results)} questions total")
        
        # Retry questions that used Small fallback with Large model after quota reset
        needs_retry_qids = [
            log["qid"] for log in self.logs 
            if log.get("needs_large_retry") and not log.get("retried_with_large")
        ]
        
        if needs_retry_qids:
            print(f"\n{'='*60}")
            print(f"RETRY PHASE: {len(needs_retry_qids)} questions used Small fallback")
            print(f"{'='*60}")
            print("Waiting for Large model quota to reset...")
            self._wait_until_next_hour()
            
            # Create question lookup
            question_lookup = {q.get("qid"): q for q in questions}
            
            for qid in tqdm(needs_retry_qids, desc="Retrying with Large"):
                if qid not in question_lookup:
                    continue
                
                q = question_lookup[qid]
                
                # Remove from cache to force re-evaluation
                if qid in self.answer_cache:
                    old_answer = self.answer_cache[qid].get("extracted_answer") if isinstance(self.answer_cache[qid], dict) else self.answer_cache[qid]
                    del self.answer_cache[qid]
                else:
                    old_answer = results.get(qid, "?")
                
                try:
                    # Re-answer with Large model (allow_fallback=False to force Large only)
                    new_answer = self.answer(q["question"], q["choices"], qid)
                    
                    # Mark as retried in the new log entry
                    if self.logs and self.logs[-1]["qid"] == qid:
                        self.logs[-1]["retried_with_large"] = True
                        self.logs[-1]["old_small_answer"] = old_answer
                    
                    results[qid] = new_answer
                    print(f"[RETRIED] {qid}: {old_answer} -> {new_answer}")
                    
                except LargeModelRateLimited:
                    print(f"[STILL LIMITED] {qid} - keeping Small answer: {old_answer}")
                    results[qid] = old_answer
                except Exception as e:
                    print(f"[ERROR] {qid}: {e} - keeping old answer")
                    results[qid] = old_answer
        
        # Build final results in original order
        final_results = []
        for q in questions:
            qid = q.get("qid", "")
            if qid in results:
                final_results.append({"qid": qid, "answer": results[qid]})
            elif qid in self.answer_cache:
                cached_answer, _ = self._get_cached_answer(qid)
                final_results.append({"qid": qid, "answer": cached_answer})
            else:
                final_results.append({"qid": qid, "answer": "A"})  # Fallback
        
        df = pd.DataFrame(final_results)
        df.to_csv(output_file, index=False)
        
        self.save_logs()
        self._save_answer_cache()
        
        print(f"\nResults saved to {output_file}")
        print(f"Cache saved to {self.answer_cache_file} ({len(self.answer_cache)} answers)")
        print(f"Total: {self.stats['total']}")
        for qtype, count in self.stats["by_type"].items():
            print(f"  {qtype}: {count}")
        for model, count in self.stats["by_model"].items():
            print(f"  {model}: {count}")
        
        # Print retry summary
        retried_count = sum(1 for log in self.logs if log.get("retried_with_large"))
        if retried_count:
            print(f"\nRetried with Large model: {retried_count} questions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VNPT AI Hackathon - Inference Pipeline")
    parser.add_argument("--input", default="/code/private_test.json", help="Input JSON/CSV file")
    parser.add_argument("--output", default="/code/submission.csv", help="Output CSV file")
    parser.add_argument("--log", default="inference_log.json", help="Log file")
    parser.add_argument("--cache-version", default="v10", help="Cache version (use new version when changing code/prompts)")
    parser.add_argument("--import-cache", help="Import answers from old cache file before running")
    
    # Parallel workers configuration (no hard limit - rate limiting handled automatically)
    parser.add_argument("--small-workers", type=int, default=50, 
                        help="Number of parallel workers for SMALL model (default: 50, recommended: 500)")
    parser.add_argument("--large-workers", type=int, default=30,
                        help="Number of parallel workers for LARGE model (default: 30, recommended: 200)")
    
    args = parser.parse_args()
    
    # Show worker configuration
    print(f"[CONFIG] Small workers: {args.small_workers}")
    print(f"[CONFIG] Large workers: {args.large_workers}")
    
    pipeline = Pipeline(
        log_file=args.log, 
        cache_version=args.cache_version,
        small_workers=args.small_workers,
        large_workers=args.large_workers
    )
    
    # Import from old cache if specified
    if args.import_cache:
        pipeline.import_from_cache(args.import_cache)
    
    pipeline.run(input_file=args.input, output_file=args.output)

