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

try:
    from build_vector_db import VectorDBSearcher
    HAS_VECTOR_DB = True
except ImportError:
    HAS_VECTOR_DB = False


class RateLimitError(Exception):
    pass


class Pipeline:
    def __init__(self, vector_db_path: str = "./data/vector_db", cache_dir: str = "./cache", log_file: str = "inference_log.json"):
        self.client = VNPTAPIClient(cache_dir=cache_dir)
        self.router = QuestionRouter()
        self.vector_db = None
        self.stats = {"total": 0, "by_type": {}, "by_model": {"small": 0, "large": 0, "none": 0}}
        self.log_file = log_file
        self.logs = []
        self.consecutive_failures = {"small": 0, "large": 0}
        self.skip_model = {"small": False, "large": False}
        
        if HAS_VECTOR_DB:
            index_path = Path(vector_db_path) / "faiss.index"
            if index_path.exists():
                try:
                    self.vector_db = VectorDBSearcher(vector_db_path)
                except Exception:
                    pass

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

    def _call_llm_with_fallback(self, messages, preferred_model: str, param_config: dict = None) -> tuple:
        if param_config is None:
            param_config = {}
        models_to_try = [preferred_model]
        if preferred_model == "small":
            models_to_try.append("large")
        elif preferred_model == "large":
            models_to_try.append("small")
        
        models_to_try = [m for m in models_to_try if not self.skip_model.get(m, False)]
        if not models_to_try:
            self.skip_model = {"small": False, "large": False}
            self.consecutive_failures = {"small": 0, "large": 0}
            models_to_try = [preferred_model, "large" if preferred_model == "small" else "small"]
        
        retry_delays = [5, 20, 40, 80, 160]
        max_retries = len(retry_delays)
        
        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    resp = self.client.chat_text(
                        messages, 
                        model=model, 
                        temperature=param_config.get("temperature", 0.3),
                        max_tokens=4096,
                        seed=param_config.get("seed", 42),
                        top_p=param_config.get("top_p", 0.9),
                        top_k=param_config.get("top_k", 10),
                        n=1
                    )
                    self.consecutive_failures[model] = 0
                    return resp, model
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate" in error_str or "limit" in error_str or "401" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            delay = retry_delays[attempt]
                            print(f"Model {model} rate limited, retry {attempt+1}/{max_retries}, waiting {delay}s...")
                            time.sleep(delay)
                            continue
                        else:
                            self.consecutive_failures[model] += 1
                            if self.consecutive_failures[model] >= 5:
                                self.skip_model[model] = True
                                print(f"Model {model} exhausted 5 times consecutively, skipping for future calls...")
                            else:
                                print(f"Model {model} exhausted after {max_retries} retries, trying next model...")
                            break
                    raise
        
        raise RateLimitError("Both models rate limited")

    def _wait_until_next_hour(self):
        from datetime import timedelta
        now = datetime.now()
        next_hour = now.replace(minute=10, second=0, microsecond=0)
        if now.minute >= 10:
            next_hour = next_hour + timedelta(hours=1)
        wait_seconds = (next_hour - now).total_seconds() + 10
        print(f"\nBoth models rate limited. Waiting until {next_hour.strftime('%H:%M')} ({wait_seconds:.0f}s)...")
        time.sleep(wait_seconds)

    def answer(self, question: str, choices: List[str], qid: str = "") -> str:
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
                if qtype == QuestionType.MATH:
                    messages1 = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp1, used_model = self._call_llm_with_fallback(messages1, preferred_model)
                    first_answer = self._extract_answer(resp1, len(choices))
                    
                    verify_prompt = f"""Một học sinh đã giải bài toán sau và chọn đáp án {first_answer}.

BÀI TOÁN:
{question}

CÁC ĐÁP ÁN:
{chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])}

LỜI GIẢI CỦA HỌC SINH:
{resp1}

NHIỆM VỤ CỦA BẠN:
1. Kiểm tra lại từng bước tính toán của học sinh
2. Xác định xem đáp án {first_answer} có đúng không
3. Nếu sai, hãy giải lại và chọn đáp án đúng
4. Nếu đúng, xác nhận đáp án

Kết thúc bằng: "Đáp án: X" (X là chữ cái cuối cùng bạn chọn)"""
                    
                    verify_messages = [
                        {"role": "system", "content": "Bạn là giám khảo Toán học với nhiệm vụ kiểm tra lại bài giải của học sinh. Hãy xác thực kết quả và đưa ra đáp án cuối cùng."},
                        {"role": "user", "content": verify_prompt}
                    ]
                    resp2, used_model = self._call_llm_with_fallback(verify_messages, preferred_model)
                    
                    responses = [resp1, resp2]
                    answer = self._extract_answer(resp2, len(choices))
                    
                    log_entry["first_answer"] = first_answer
                    log_entry["verification"] = True
                    
                elif qtype == QuestionType.SAFETY:
                    messages = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp, used_model = self._call_llm_with_fallback(messages, preferred_model)
                    responses = [resp]
                    answer = self._extract_answer(resp, len(choices))
                    
                elif qtype == QuestionType.READING and preferred_model == "large":
                    messages1 = self.router.build_prompt(qtype, question, choices, context, 0)
                    resp1, used_model = self._call_llm_with_fallback(messages1, preferred_model)
                    first_answer = self._extract_answer(resp1, len(choices))
                    
                    verify_prompt = f"""Một học sinh đã trả lời câu hỏi đọc hiểu và chọn đáp án {first_answer}.

CÂU HỎI VÀ ĐOẠN VĂN:
{question}

CÁC ĐÁP ÁN:
{chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])}

LỜI GIẢI CỦA HỌC SINH:
{resp1}

NHIỆM VỤ:
1. Kiểm tra lại - đáp án {first_answer} có được hỗ trợ bởi văn bản không?
2. Tìm bằng chứng TRỰC TIẾP trong văn bản
3. Nếu sai, chọn đáp án đúng hơn
4. Nếu đúng, xác nhận

Kết thúc: "Đáp án: X" """
                    
                    verify_messages = [
                        {"role": "system", "content": "Bạn là giám khảo đọc hiểu. Kiểm tra bài làm và xác nhận hoặc sửa đáp án."},
                        {"role": "user", "content": verify_prompt}
                    ]
                    resp2, used_model = self._call_llm_with_fallback(verify_messages, preferred_model)
                    
                    responses = [resp1, resp2]
                    answer = self._extract_answer(resp2, len(choices))
                    log_entry["first_answer"] = first_answer
                    log_entry["verification"] = True
                    
                else:
                    param_configs = [
                        {"temperature": 0.1, "top_p": 0.7, "top_k": 10, "seed": 42},
                        {"temperature": 0.5, "top_p": 0.85, "top_k": 10, "seed": 123},
                        {"temperature": 1.0, "top_p": 1.0, "top_k": 10, "seed": 456},
                    ]
                    
                    responses = []
                    used_model = None
                    for prompt_idx in range(3):
                        messages = self.router.build_prompt(qtype, question, choices, context, prompt_idx)
                        try:
                            resp, used_model = self._call_llm_with_fallback(messages, preferred_model, param_configs[prompt_idx])
                            responses.append(resp)
                        except Exception as prompt_error:
                            if "rate" in str(prompt_error).lower() or "limit" in str(prompt_error).lower():
                                raise prompt_error
                            responses.append(None)
                    
                    responses = [r for r in responses if r is not None]
                    if not responses:
                        raise Exception("All prompt calls failed")
                    
                    refusal_patterns = ["tôi không thể", "không thể trả lời", "không thể cung cấp", "từ chối"]
                    votes = []
                    for resp in responses:
                        is_refusal = any(p in resp.lower() for p in refusal_patterns)
                        if is_refusal and safe_idx is not None:
                            votes.append(chr(65 + safe_idx))
                        else:
                            votes.append(self._extract_answer(resp, len(choices)))
                    
                    from collections import Counter
                    vote_counts = Counter(votes)
                    top_vote = vote_counts.most_common(1)[0]
                    
                    if top_vote[1] == 1 and len(votes) == 3:
                        log_entry["conflict"] = True
                        tiebreak_prompt = f"""Ba chuyên gia đã phân tích câu hỏi này và đưa ra đáp án khác nhau:
- Chuyên gia 1 chọn: {votes[0]}
- Chuyên gia 2 chọn: {votes[1]}  
- Chuyên gia 3 chọn: {votes[2]}

CÂU HỎI:
{question}

CÁC ĐÁP ÁN:
{chr(10).join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])}

Hãy phân tích cẩn thận và chọn đáp án ĐÚNG NHẤT.
Kết thúc: "Đáp án: X" """
                        
                        tiebreak_messages = [
                            {"role": "system", "content": "Bạn là trọng tài giải quyết tranh cãi. Phân tích kỹ và chọn đáp án đúng nhất."},
                            {"role": "user", "content": tiebreak_prompt}
                        ]
                        tiebreak_resp, used_model = self._call_llm_with_fallback(tiebreak_messages, preferred_model)
                        responses.append(tiebreak_resp)
                        answer = self._extract_answer(tiebreak_resp, len(choices))
                    else:
                        answer = top_vote[0]
                    
                    log_entry["votes"] = votes
                    log_entry["vote_counts"] = dict(vote_counts)
                
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
                
                if "không thể trả lời" in error_str.lower() or "khong the tra loi" in error_str.lower():
                    answer = self._find_cannot_answer_choice(choices)
                    log_entry["extracted_answer"] = answer
                    log_entry["reasoning"] = "LLM refused - selected cannot answer option"
                else:
                    answer = "A"
                    log_entry["extracted_answer"] = answer
                break
        
        self.logs.append(log_entry)
        return answer

    def _extract_answer(self, text: str, num_choices: int) -> str:
        valid = [chr(65 + i) for i in range(num_choices)]
        
        patterns = [
            r'[Đđ][áa]p\s*[áa]n[:\s]*\*?\*?([A-Ja-j])',
            r'[Đđ][áa]p\s*[áa]n\s+[đd][úu]ng[:\s]*\*?\*?([A-Ja-j])',
            r'[Đđ][áa]p\s*[áa]n\s+l[àa][:\s]*\*?\*?([A-Ja-j])',
            r'[Cc]h[oọ]n[:\s]*\*?\*?([A-Ja-j])',
            r'[Kk][êế]t\s*lu[âậ]n[:\s]*\*?\*?([A-Ja-j])',
            r'\*\*([A-Ja-j])\*\*\s*$',
            r'→\s*\*?\*?([A-Ja-j])[.\s\)]',
            r'l[àa]\s+\*?\*?([A-Ja-j])[.\s\)]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                ans = matches[-1].upper()
                if ans in valid:
                    return ans
        
        lines = text.strip().split('\n')
        for line in reversed(lines[-15:]):
            line_upper = line.upper()
            if 'ĐÁP ÁN' in line.upper() or 'DAP AN' in line_upper:
                for v in reversed(valid):
                    if v in line_upper:
                        return v
        
        for line in reversed(lines[-10:]):
            for v in valid:
                if re.search(rf'\b{v}[.\s\)]', line):
                    return v
        
        return "A"

    def _find_cannot_answer_choice(self, choices: List[str]) -> str:
        cannot_answer_patterns = [
            r'không thể trả lời',
            r'khong the tra loi',
            r'không trả lời được',
            r'từ chối trả lời',
            r'không cung cấp',
            r'không hỗ trợ',
        ]
        
        for idx, choice in enumerate(choices):
            choice_lower = choice.lower()
            for pattern in cannot_answer_patterns:
                if re.search(pattern, choice_lower):
                    return chr(65 + idx)
        
        return "A"

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
        
        results = []
        for q in tqdm(questions, desc="Processing"):
            answer = self.answer(q["question"], q["choices"], q.get("qid", ""))
            results.append({"qid": q["qid"], "answer": answer})
            self.stats["total"] += 1
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        self.save_logs()
        
        print(f"\nResults saved to {output_file}")
        print(f"Total: {self.stats['total']}")
        for qtype, count in self.stats["by_type"].items():
            print(f"  {qtype}: {count}")
        for model, count in self.stats["by_model"].items():
            print(f"  {model}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/code/private_test.json", help="Input JSON file")
    parser.add_argument("--output", default="submission.csv", help="Output CSV file")
    parser.add_argument("--log", default="inference_log.json", help="Log file")
    args = parser.parse_args()
    
    pipeline = Pipeline(log_file=args.log)
    pipeline.run(input_file=args.input, output_file=args.output)
