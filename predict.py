import json
import hashlib
import re
import time
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm

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
    def __init__(self, vector_db_path: str = "./data/vector_db", cache_dir: str = "./cache", log_file: str = "inference_log.json", cache_version: str = "v9"):
        self.client = VNPTAPIClient(cache_dir=cache_dir)
        self.router = QuestionRouter()
        self.vector_db = None
        self.stats = {"total": 0, "by_type": {}, "by_model": {"small": 0, "large": 0, "none": 0}}
        self.log_file = log_file
        self.logs = []
        self.consecutive_failures = {"small": 0, "large": 0}
        self.skip_model = {"small": False, "large": False}
        
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
        """Save answer cache to file with metadata"""
        data = {
            "version": self.cache_version,
            "count": len(self.answer_cache),
            "answers": self.answer_cache
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
        Call LLM with optional fallback from Large to Small.
        Returns: (response, used_model, fallback_used)
        - fallback_used: True if Large was rate limited and fell back to Small
        """
        if param_config is None:
            param_config = {}
        
        model = preferred_model
        retry_delays = [5, 10, 20, 40]
        max_retries = len(retry_delays)
        last_error = None
        
        for attempt in range(max_retries):
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
                self.consecutive_failures[model] = 0
                return resp, model, False  # No fallback used
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                is_rate_limit = "rate" in error_str or "limit" in error_str or "401" in error_str or "429" in error_str
                is_server_error = "invalid response" in error_str or "datasign" in error_str or "500" in error_str
                is_content_filtered = "content filtered" in error_str
                
                # Content filtered - don't retry
                if is_content_filtered:
                    raise
                
                if attempt == 0:
                    print(f"[DEBUG] Model {model} error: rate_limit={is_rate_limit}, server={is_server_error}")
                
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        print(f"Model {model} rate limited, retry {attempt+1}/{max_retries}, waiting {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        # Rate limit exhausted
                        if model == "large" and allow_fallback:
                            # Fallback to Small model, mark for retry later
                            print(f"[LARGE RATE LIMITED] Falling back to Small model...")
                            try:
                                resp = self.client.chat_text(
                                    messages, 
                                    model="small", 
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192,
                                    seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9),
                                    top_k=param_config.get("top_k", 10),
                                    n=1
                                )
                                return resp, "small", True  # Fallback was used
                            except Exception as small_error:
                                raise RateLimitError(f"Both models failed: Large={last_error}, Small={small_error}")
                        elif model == "large":
                            raise LargeModelRateLimited(f"Large model rate limited after {max_retries} retries")
                        else:
                            raise RateLimitError(f"Small model exhausted: {last_error}")
                
                elif is_server_error:
                    if attempt < 2:
                        wait_time = 60 * (attempt + 1)
                        print(f"Model {model} server error, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if model == "large" and allow_fallback:
                            print(f"[LARGE SERVER ERROR] Falling back to Small model...")
                            try:
                                resp = self.client.chat_text(
                                    messages, 
                                    model="small", 
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192,
                                    seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9),
                                    top_k=param_config.get("top_k", 10),
                                    n=1
                                )
                                return resp, "small", True
                            except:
                                raise LargeModelRateLimited(f"Large model server error: {last_error}")
                        elif model == "large":
                            raise LargeModelRateLimited(f"Large model server error: {last_error}")
                        else:
                            raise RateLimitError(f"Small model server error: {last_error}")
                
                else:
                    # Unknown error
                    if attempt < 2:
                        print(f"Model {model} error: {str(e)[:80]}, retry {attempt+1}/3...")
                        time.sleep(5)
                        continue
                    else:
                        if model == "large" and allow_fallback:
                            print(f"[LARGE FAILED] Falling back to Small model...")
                            try:
                                resp = self.client.chat_text(
                                    messages, 
                                    model="small", 
                                    temperature=param_config.get("temperature", 0.8),
                                    max_tokens=8192,
                                    seed=param_config.get("seed", 42),
                                    top_p=param_config.get("top_p", 0.9),
                                    top_k=param_config.get("top_k", 10),
                                    n=1
                                )
                                return resp, "small", True
                            except:
                                raise LargeModelRateLimited(f"Large model failed: {last_error}")
                        elif model == "large":
                            raise LargeModelRateLimited(f"Large model failed: {last_error}")
                        else:
                            raise RateLimitError(f"Small model failed: {last_error}")
        
        # Should not reach here
        raise RateLimitError(f"Model {model} failed: {last_error}")

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
        
        context = self.retrieve(question, k=5) if meta.get("use_rag") else None
        
        first_prompt = self.router.build_prompt(qtype, question, choices, context, 0)
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
                    # READING: 4-call voting (2 prompts x 2 models when Large fails)
                    choices_str = chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    votes = []
                    fallback_used = False
                    
                    # Call 1: 5-step deep analysis prompt with Large
                    messages1 = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp1, model1, fb1 = self._call_llm_with_fallback(messages1, "large",
                        {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 42})
                    answer1 = self._extract_answer(resp1, len(choices))
                    votes.append({"answer": answer1, "model": model1, "prompt": "v1"})
                    if fb1:
                        fallback_used = True
                    
                    # Call 2: 4-step data extraction prompt with Large
                    messages2 = self.router._build_reading_prompt_v2(question, choices_str)
                    resp2, model2, fb2 = self._call_llm_with_fallback(messages2, "large",
                        {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 123})
                    answer2 = self._extract_answer(resp2, len(choices))
                    votes.append({"answer": answer2, "model": model2, "prompt": "v2"})
                    if fb2:
                        fallback_used = True
                    
                    responses = [resp1, resp2]
                    log_entry["reading_votes"] = [v["answer"] for v in votes]
                    log_entry["reading_models"] = [v["model"] for v in votes]
                    log_entry["fallback_used"] = fallback_used
                    
                    # Count votes
                    vote_counts = {}
                    for v in votes:
                        ans = v["answer"]
                        vote_counts[ans] = vote_counts.get(ans, 0) + 1
                    
                    # Find majority answer
                    max_votes = max(vote_counts.values())
                    majority_answers = [ans for ans, count in vote_counts.items() if count == max_votes]
                    
                    if len(majority_answers) == 1:
                        # Clear winner
                        answer = majority_answers[0]
                        log_entry["reading_consensus"] = True
                    else:
                        # Tie - use tiebreak with Large (or Small if rate limited)
                        choice_a_text = choices[ord(answer1) - 65] if answer1 and ord(answer1) - 65 < len(choices) else ""
                        choice_b_text = choices[ord(answer2) - 65] if answer2 and ord(answer2) - 65 < len(choices) else ""
                        
                        tiebreak_prompt = f"""HAI CHUYÊN GIA ĐỌC HIỂU ĐƯA RA 2 ĐÁP ÁN KHÁC NHAU:

Chuyên gia 1 chọn: {answer1}. {choice_a_text}
Lý do: {resp1[:1000] if resp1 else ""}

Chuyên gia 2 chọn: {answer2}. {choice_b_text}  
Lý do: {resp2[:1000] if resp2 else ""}

VĂN BẢN VÀ CÂU HỎI GỐC:
{question}

CÁC ĐÁP ÁN:
{choices_str}

NHIỆM VỤ CỦA BẠN:
1. Đọc lại văn bản gốc một cách cẩn thận
2. Xem xét lập luận của cả 2 chuyên gia
3. Xác định đáp án nào được HỖ TRỢ TRỰC TIẾP hơn bởi văn bản
4. Nếu cả 2 đều sai, chọn đáp án đúng nhất

Luôn kết thúc bằng: "Đáp án cuối cùng: X" (X là chữ cái cuối cùng bạn chọn)"""

                        tiebreak_messages = [
                            {"role": "system", "content": "Bạn là giám khảo cao cấp với nhiệm vụ phân xử khi có bất đồng. Hãy đọc kỹ văn bản và chọn đáp án được HỖ TRỢ RÕ RÀNG NHẤT."},
                            {"role": "user", "content": tiebreak_prompt}
                        ]
                        
                        try:
                            resp3, model3, fb3 = self._call_llm_with_fallback(tiebreak_messages, "large",
                                {"temperature": 0.1, "top_p": 0.9, "top_k": 5, "seed": 999})
                            responses.append(resp3)
                            answer = self._extract_answer(resp3, len(choices))
                            log_entry["tiebreak_answer"] = answer
                            log_entry["tiebreak_model"] = model3
                            if fb3:
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
                if attempt < max_attempts - 1:
                    self._wait_until_next_hour()
                else:
                    log_entry["error"] = "Rate limit exceeded after all attempts"
                    answer = "A"
                    log_entry["extracted_answer"] = answer
                    
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
        valid = [chr(65 + i) for i in range(num_choices)]
        
        lines = text.strip().split('\n')
        
        # PRIORITY 1: Look for "Đáp án cuối cùng" - highest priority
        for line in reversed(lines[-20:]):
            # Match "Đáp án cuối cùng: X" or "Đáp án cuối cùng là X"
            match = re.search(r'[Đđ][áa]p\s*[áa]n\s*[Cc]u[ôố]i\s*[Cc]ùng[:\s]*\*?\*?\s*([A-Ja-j])', line, re.IGNORECASE)
            if match:
                ans = match.group(1).upper()
                if ans in valid:
                    return ans
        
        # PRIORITY 2: Look for bold answer at end - "**Đáp án: X**" or "✅ Đáp án: X"
        final_patterns = [
            r'✅\s*\*?\*?[Đđ][áa]p\s*[áa]n[:\s]*\*?\*?\s*([A-Ja-j])',
            r'\*\*\s*[Đđ][áa]p\s*[áa]n[:\s]*([A-Ja-j])\s*\*\*',
            r'[Đđ][áa]p\s*[áa]n[:\s]*\*\*\s*([A-Ja-j])\s*\*\*',
        ]
        for line in reversed(lines[-15:]):
            for pattern in final_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    ans = match.group(1).upper()
                    if ans in valid:
                        return ans
        
        # PRIORITY 3: Look for explicit answer patterns in last 15 lines
        answer_patterns = [
            r'[Đđ][áa]p\s*[áa]n[:\s]+\*?\*?([A-Ja-j])(?:[.\s\)\*\}]|$)',
            r'[Đđ][áa]p\s*[áa]n\s+[đd][úu]ng[:\s]+\*?\*?([A-Ja-j])',
            r'[Đđ][áa]p\s*[áa]n\s+l[àa][:\s]+\*?\*?([A-Ja-j])',
            r'[Kk][êế]t\s*lu[âậ]n[:\s]+.*[Đđ][áa]p\s*[áa]n[:\s]*\*?\*?([A-Ja-j])',
            r'[Cc]h[oọ]n\s+[Đđ][áa]p[:\s]*([A-Ja-j])',
            r'SELECTED\s+ANSWER[:\s]*([A-Ja-j])',
        ]
        for line in reversed(lines[-15:]):
            for pattern in answer_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    ans = match.group(1).upper()
                    if ans in valid:
                        return ans
        
        # PRIORITY 4: Look for standalone "Đáp án: X" in whole text (last match)
        all_matches = re.findall(r'[Đđ][áa]p\s*[áa]n[:\s]+\*?\*?([A-Ja-j])', text, re.IGNORECASE)
        if all_matches:
            ans = all_matches[-1].upper()
            if ans in valid:
                return ans
        
        # PRIORITY 5: Look for bold letter at very end like "**B**" on last few lines
        for line in reversed(lines[-5:]):
            match = re.search(r'\*\*\s*([A-Ja-j])\s*\*\*\s*$', line)
            if match:
                ans = match.group(1).upper()
                if ans in valid:
                    return ans
        
        # PRIORITY 6: Fallback - find any "Đáp án" mention in text
        for line in reversed(lines):
            line_upper = line.upper()
            if 'ĐÁP ÁN' in line_upper or 'DAP AN' in line_upper:
                for v in valid:
                    if f': {v}' in line_upper or f'= {v}' in line_upper or re.search(rf'\b{v}\b', line_upper):
                        return v
        
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

    def save_logs(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"Logs saved to {self.log_file}")

    def run(self, input_file: str = "/code/private_test.json", output_file: str = "submission.csv"):
        print("=" * 60)
        print("VNPT AI HACKATHON - INFERENCE")
        print("=" * 60)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        print(f"Loaded {len(questions)} questions")
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
        pending_large = []  # Questions deferred due to rate limit
        
        # Process all questions - Small first, then Large
        all_pending = small_questions + large_questions
        
        for q in tqdm(all_pending, desc="Processing"):
            qid = q.get("qid", "")
            
            # Re-check cache (may have been filled during run)
            if qid in self.answer_cache:
                cached_answer, _ = self._get_cached_answer(qid)
                results[qid] = cached_answer
                continue
            
            try:
                answer = self.answer(q["question"], q["choices"], qid)
                results[qid] = answer
                self.stats["total"] += 1
            except LargeModelRateLimited as e:
                # Large model rate limited - defer this question
                print(f"[DEFERRED] {qid} - Large model rate limited, will retry later")
                pending_large.append(q)
            except RateLimitError as e:
                # Small model rate limited - wait and retry
                print(f"\n[SMALL MODEL RATE LIMITED] {qid} - waiting for quota reset...")
                self._wait_until_next_hour()
                try:
                    answer = self.answer(q["question"], q["choices"], qid)
                    results[qid] = answer
                    self.stats["total"] += 1
                except Exception as e2:
                    print(f"[ERROR] {qid} still failed after wait: {e2}")
                    results[qid] = "A"  # Fallback
        
        # Process pending Large questions (after quota reset)
        if pending_large:
            print(f"\n{'='*60}")
            print(f"Processing {len(pending_large)} deferred Large model questions...")
            print(f"{'='*60}")
            
            # Wait for quota reset before retrying Large questions
            print("Waiting for Large model quota to reset...")
            self._wait_until_next_hour()
            
            for q in tqdm(pending_large, desc="Deferred Large"):
                qid = q.get("qid", "")
                
                if qid in self.answer_cache:
                    cached_answer, _ = self._get_cached_answer(qid)
                    results[qid] = cached_answer
                    continue
                
                try:
                    answer = self.answer(q["question"], q["choices"], qid)
                    results[qid] = answer
                    self.stats["total"] += 1
                except LargeModelRateLimited:
                    # After 5 fails, skip and add to retry queue
                    print(f"[DEFERRED AGAIN] {qid} - adding to retry queue for next hour")
                    pending_large.append(q)  # Re-add to pending queue
                            
                except Exception as e:
                    print(f"[ERROR] {qid}: {e}")
                    results[qid] = "A"
        
        # Round-robin retry loop: keep retrying until all pending_large are done
        while pending_large:
            failed_this_round = []
            
            print(f"\n{'='*60}")
            print(f"Waiting for next hour to retry {len(pending_large)} pending questions...")
            print(f"{'='*60}")
            self._wait_until_next_hour()
            
            for q in tqdm(pending_large, desc="Retry Round"):
                qid = q.get("qid", "")
                
                if qid in self.answer_cache:
                    cached_answer, _ = self._get_cached_answer(qid)
                    results[qid] = cached_answer
                    continue
                
                if qid in results:
                    continue  # Already solved
                
                try:
                    answer = self.answer(q["question"], q["choices"], qid)
                    results[qid] = answer
                    self.stats["total"] += 1
                    print(f"[SUCCESS] {qid} -> {answer}")
                except LargeModelRateLimited:
                    print(f"[STILL RATE LIMITED] {qid} - will retry next hour")
                    failed_this_round.append(q)
                except Exception as e:
                    print(f"[ERROR] {qid}: {e} - will retry next hour")
                    failed_this_round.append(q)
            
            pending_large = failed_this_round
            if pending_large:
                print(f"\n{len(pending_large)} questions still pending, will retry next hour...")
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/code/private_test.json", help="Input JSON file")
    parser.add_argument("--output", default="submission.csv", help="Output CSV file")
    parser.add_argument("--log", default="inference_log.json", help="Log file")
    parser.add_argument("--cache-version", default="v4", help="Cache version (use new version when changing code/prompts)")
    parser.add_argument("--import-cache", help="Import answers from old cache file before running")
    args = parser.parse_args()
    
    pipeline = Pipeline(log_file=args.log, cache_version=args.cache_version)
    
    # Import from old cache if specified
    if args.import_cache:
        pipeline.import_from_cache(args.import_cache)
    
    pipeline.run(input_file=args.input, output_file=args.output)
