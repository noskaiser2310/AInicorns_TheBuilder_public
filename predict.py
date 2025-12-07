import json
import hashlib
import re
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


class Pipeline:
    def __init__(self, vector_db_path: str = "./data/vector_db", cache_dir: str = "./cache", log_file: str = "inference_log.json"):
        self.client = VNPTAPIClient(cache_dir=cache_dir)
        self.router = QuestionRouter()
        self.vector_db = None
        self.stats = {"total": 0, "by_type": {}, "by_model": {"small": 0, "large": 0, "none": 0}}
        self.log_file = log_file
        self.logs = []
        
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
        
        if qtype == QuestionType.SAFETY and meta.get("safe_idx") is not None:
            self.stats["by_model"]["none"] += 1
            answer = chr(65 + meta["safe_idx"])
            log_entry["answer"] = answer
            log_entry["reasoning"] = "Safety rule-based answer"
            self.logs.append(log_entry)
            return answer
        
        context = self.retrieve(question, k=5) if meta.get("use_rag") else None
        messages = self.router.build_prompt(qtype, question, choices, context)
        
        log_entry["prompt_system"] = messages[0]["content"][:300] + "..."
        log_entry["prompt_user"] = messages[1]["content"][:500] + "..."
        
        if model_choice == ModelChoice.NONE:
            self.stats["by_model"]["none"] += 1
            log_entry["answer"] = "A"
            log_entry["reasoning"] = "Fallback to A"
            self.logs.append(log_entry)
            return "A"
        
        model = model_choice.value
        self.stats["by_model"][model] += 1
        
        try:
            import time
            answer_text = self.client.chat_text(
                messages, 
                model=model, 
                temperature=0.2,
                max_tokens=2000,
                seed=42
            )
            time.sleep(1)
            
            log_entry["model_response"] = answer_text
            answer = self._extract_answer(answer_text, len(choices))
            log_entry["extracted_answer"] = answer
            
        except Exception as e:
            log_entry["error"] = str(e)
            answer = "A"
            log_entry["extracted_answer"] = answer
        
        self.logs.append(log_entry)
        return answer

    def _extract_answer(self, text: str, num_choices: int) -> str:
        valid = [chr(65 + i) for i in range(num_choices)]
        
        patterns = [
            r'[Đđ]áp\s*[aá]n[:\s]+([A-Za-z])\b',
            r'[Đđ]áp\s*[aá]n\s+đúng[:\s]+([A-Za-z])\b',
            r'[Đđ]áp\s*[aá]n\s+là[:\s]+([A-Za-z])\b',
            r'[Cc]họn[:\s]+([A-Za-z])\b',
            r'[Kk]ết\s*luận[:\s]+([A-Za-z])\b',
            r'\*\*([A-Za-z])\*\*',
            r'là\s+([A-Za-z])[.\s\)]',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                last_match = matches[-1].group(1).upper()
                if last_match in valid:
                    return last_match
        
        last_lines = text.strip().split('\n')[-10:]
        for line in reversed(last_lines):
            line_clean = line.strip()
            for ans in valid:
                if f"Đáp án: {ans}" in line_clean or f"Đáp án {ans}" in line_clean:
                    return ans
                if f"đáp án: {ans}" in line_clean.lower() or f"đáp án {ans}" in line_clean.lower():
                    return ans
                if line_clean.upper() == ans or line_clean.upper().startswith(ans + '.'):
                    return ans
        
        for ans in reversed(valid):
            if f" {ans}." in text or f" {ans})" in text or f"**{ans}**" in text:
                return ans
        
        return valid[0]

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
