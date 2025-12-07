import json
import argparse
from collections import defaultdict
from tqdm import tqdm

from predict import Pipeline
from question_router import QuestionRouter


def evaluate(val_path: str = "data/val.json", max_questions: int = None, output_file: str = None):
    print("=" * 60)
    print("EVALUATION WITH LOGGING")
    print("=" * 60)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_questions:
        data = data[:max_questions]
    
    print(f"Loaded {len(data)} questions")
    
    log_file = output_file.replace('.json', '_logs.json') if output_file else "eval_logs.json"
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
        print("\nSample errors:")
        for err in errors[:5]:
            print(f"  {err['qid']}: predicted={err['predicted']}, expected={err['expected']} ({err['type']})")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": accuracy,
                "by_type": {k: {"correct": correct_by_type[k], "total": total_by_type[k]} for k in total_by_type},
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
        print(f"Logs saved to {log_file}")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", default="data/val.json")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()
    
    evaluate(val_path=args.val_path, max_questions=args.max, output_file=args.output)
