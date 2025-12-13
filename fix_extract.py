# -*- coding: utf-8 -*-
"""
STRICT extraction: Answer MUST be directly after "Đáp án cuối cùng"
Either same line OR next line, but MUST be anchored to this phrase
"""
import json
import re

def extract_answer_strict(text: str, num_choices: int) -> str:
    """ONLY extract answer that appears RIGHT AFTER 'Đáp án cuối cùng'"""
    valid = [chr(65 + i) for i in range(num_choices)]
    lines = text.strip().split('\n')
    
    found_answers = []
    
    for i, line in enumerate(lines):
        # Check if this line contains "Đáp án cuối cùng"
        if 'đáp án cuối cùng' in line.lower():
            # Case 1: Answer on SAME line - "Đáp án cuối cùng: A" or "Đáp án cuối cùng là A"
            match = re.search(r'(?:ĐÁP ÁN CUỐI CÙNG|Đáp án cuối cùng)[:\s]*(?:là)?[\s\*]*\[?([A-Ja-j])[\.\]\s\*\)\,]?', line, re.IGNORECASE)
            if match:
                ans = match.group(1).upper()
                if ans in valid:
                    found_answers.append(ans)
                    continue  # Found on same line, move to next occurrence
            
            # Case 2: Answer on NEXT line (only if same line didn't have it)
            # Format: "**A. text**" or "**A**" at START of next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                match = re.search(r'^\*\*\s*([A-Ja-j])[\.\s\*\)]', next_line)
                if match:
                    ans = match.group(1).upper()
                    if ans in valid:
                        found_answers.append(ans)
    
    # Return the LAST found answer (most recent = final answer)
    if found_answers:
        return found_answers[-1]
    
    return None  # No match found


# Load cache
with open("answer_cache_v11.json", "r", encoding="utf-8") as f:
    cache = json.load(f)

# Check ALL questions, not just STEM
changes = []

for qid, data in cache.get("answers", {}).items():
    responses = data.get("model_responses", [])
    if not responses:
        continue
    
    # Check LAST response
    last_response = responses[-1]
    num_choices = len(data.get("choices", [])) or 4
    
    old_answer = data.get("extracted_answer", "")
    new_answer = extract_answer_strict(last_response, num_choices)
    
    # Only change if new answer is found AND different
    if new_answer and new_answer != old_answer:
        changes.append({
            "qid": qid,
            "type": data.get("type", ""),
            "old": old_answer,
            "new": new_answer,
            "snippet": last_response[-200:]
        })

# Show changes
print(f"Found {len(changes)} potential fixes:\n")
for c in changes:
    print(f"{c['qid']} ({c['type']}): {c['old']} -> {c['new']}")
    print(f"  ...{c['snippet'][-100:]}")
    print()

if changes:
    print("\n" + "="*60)
    apply = input("Apply these changes? (y/n): ").strip().lower()
    if apply == 'y':
        for c in changes:
            cache["answers"][c["qid"]]["extracted_answer"] = c["new"]
        with open("answer_cache_v11.json", "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Applied {len(changes)} fixes!")
    else:
        print("Cancelled.")
else:
    print("No fixes needed - all answers already correct!")
