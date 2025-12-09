import json
import argparse
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from predict import Pipeline
from question_router import QuestionRouter


def evaluate(val_path: str = "data/val.json", max_questions: int = None, 
             start: int = None, end: int = None, output_file: str = None):
    print("=" * 60)
    print("EVALUATION WITH LOGGING")
    print("=" * 60)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    
    if start is not None and end is not None:
        data = data[start:end+1]
        print(f"Selected questions {start} to {end} (total: {len(data)})")
    elif start is not None:
        data = data[start:]
        print(f"Selected questions from {start} onwards (total: {len(data)})")
    elif max_questions:
        data = data[:max_questions]
        print(f"Selected first {max_questions} questions")
    
    print(f"Loaded {len(data)} questions from {total_questions} total")
    
    os.makedirs("eval", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    range_str = f"_{start}-{end}" if start is not None else (f"_first{max_questions}" if max_questions else "_all")
    
    if output_file:
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        result_file = f"eval/{base_name}{range_str}_{timestamp}.json"
        log_file = f"eval/{base_name}{range_str}_{timestamp}_logs.json"
    else:
        result_file = f"eval/eval{range_str}_{timestamp}.json"
        log_file = f"eval/eval{range_str}_{timestamp}_logs.json"
    
    pipeline = Pipeline(log_file=log_file)
    router = QuestionRouter()
    
    results = []
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    
    for q in tqdm(data, desc="Evaluating"):
        qid = q["qid"]
        question = q["question"]
        choices = q["choices"]
        expected = q.get("answer", "").upper()
        
        qtype, _, _ = router.classify(question, choices)
        predicted = pipeline.answer(question, choices, qid)
        
        is_correct = predicted == expected
        results.append({
            "qid": qid,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "type": qtype.value
        })
        
        total_by_type[qtype.value] += 1
        if is_correct:
            correct_by_type[qtype.value] += 1
    
    pipeline.save_logs()
    
    total_correct = sum(correct_by_type.values())
    total = len(results)
    accuracy = total_correct / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    
    for qtype in sorted(total_by_type.keys()):
        correct = correct_by_type[qtype]
        total_t = total_by_type[qtype]
        acc = correct / total_t * 100 if total_t > 0 else 0
        print(f"{qtype:<20} {correct:>8} {total_t:>8} {acc:>9.1f}%")
    
    print("-" * 50)
    print(f"{'OVERALL':<20} {total_correct:>8} {total:>8} {accuracy:>9.1f}%")
    
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nErrors ({len(errors)} total):")
        for err in errors[:10]:
            print(f"  {err['qid']}: predicted={err['predicted']}, expected={err['expected']} ({err['type']})")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total": total,
            "range": {"start": start, "end": end, "max": max_questions},
            "by_type": {k: {"correct": correct_by_type[k], "total": total_by_type[k]} for k in total_by_type},
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print(f"Logs saved to {log_file}")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument("--val-path", default="data/val.json", help="Path to validation JSON")
    parser.add_argument("--max", type=int, default=None, help="Max number of questions (from start)")
    parser.add_argument("--start", type=int, default=None, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (inclusive)")
    parser.add_argument("--output", default=None, help="Output file base name")
    args = parser.parse_args()
    
    evaluate(val_path=args.val_path, max_questions=args.max, 
             start=args.start, end=args.end, output_file=args.output)
