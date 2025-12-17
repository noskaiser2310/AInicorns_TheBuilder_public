"""
Evaluation Script for VNPT AI Hackathon
Usage: python evaluate.py --pred <prediction_file> --gt <ground_truth_file>

Prediction file: JSON with format {"answers": {"qid": {"extracted_answer": "A", ...}, ...}}
                 or cache file format
Ground truth file: JSON with format [{"qid": "val_001", "answer": "A", ...}, ...]
"""

import json
import argparse
from collections import defaultdict


def load_predictions(filepath: str) -> dict:
    """Load predictions from cache file or submission-style JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {}
    
    # Format 1: Cache file with "answers" key
    if isinstance(data, dict) and "answers" in data:
        for qid, entry in data["answers"].items():
            if isinstance(entry, dict):
                predictions[qid] = entry.get("extracted_answer", entry.get("answer", ""))
            else:
                predictions[qid] = str(entry)
    
    # Format 2: List of {"qid": ..., "answer": ...}
    elif isinstance(data, list):
        for item in data:
            qid = item.get("qid", "")
            ans = item.get("answer", item.get("extracted_answer", ""))
            if qid:
                predictions[qid] = ans
    
    # Format 3: Simple dict {qid: answer}
    elif isinstance(data, dict):
        for qid, entry in data.items():
            if isinstance(entry, dict):
                predictions[qid] = entry.get("extracted_answer", entry.get("answer", ""))
            else:
                predictions[qid] = str(entry)
    
    return predictions


def load_ground_truth(filepath: str) -> dict:
    """Load ground truth from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt = {}
    
    # Format 1: List of {"qid": ..., "answer": ...}
    if isinstance(data, list):
        for item in data:
            qid = item.get("qid", "")
            ans = item.get("answer", "")
            if qid and ans:
                gt[qid] = ans
    
    # Format 2: Dict {qid: answer} or {qid: {"answer": ...}}
    elif isinstance(data, dict):
        for qid, entry in data.items():
            if isinstance(entry, dict):
                gt[qid] = entry.get("answer", "")
            else:
                gt[qid] = str(entry)
    
    return gt


def evaluate(predictions: dict, ground_truth: dict, verbose: bool = True) -> dict:
    """Compare predictions with ground truth and return metrics"""
    
    correct = 0
    wrong = 0
    not_found = 0
    
    wrong_list = []
    by_type = defaultdict(lambda: {"correct": 0, "wrong": 0})
    
    for qid, gt_ans in ground_truth.items():
        if qid in predictions:
            pred_ans = predictions[qid]
            if pred_ans == gt_ans:
                correct += 1
            else:
                wrong += 1
                wrong_list.append({
                    "qid": qid,
                    "predicted": pred_ans,
                    "correct": gt_ans
                })
        else:
            not_found += 1
    
    total = correct + wrong
    accuracy = correct / total * 100 if total > 0 else 0
    
    results = {
        "total_gt": len(ground_truth),
        "matched": total,
        "correct": correct,
        "wrong": wrong,
        "not_found": not_found,
        "accuracy": accuracy,
        "wrong_list": wrong_list
    }
    
    if verbose:
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Ground truth questions: {results['total_gt']}")
        print(f"Matched in predictions: {results['matched']}")
        print(f"Correct: {results['correct']}")
        print(f"Wrong: {results['wrong']}")
        print(f"Not found in predictions: {results['not_found']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        
        if wrong_list:
            print()
            print("=== WRONG ANSWERS ===")
            for item in wrong_list:
                print(f"  {item['qid']}: pred={item['predicted']} | correct={item['correct']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth")
    parser.add_argument("--pred", required=True, help="Prediction file (cache JSON or submission)")
    parser.add_argument("--gt", required=True, help="Ground truth file (val JSON)")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from: {args.pred}")
    predictions = load_predictions(args.pred)
    print(f"  -> {len(predictions)} predictions loaded")
    
    print(f"Loading ground truth from: {args.gt}")
    ground_truth = load_ground_truth(args.gt)
    print(f"  -> {len(ground_truth)} ground truth answers loaded")
    print()
    
    results = evaluate(predictions, ground_truth, verbose=not args.quiet)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
