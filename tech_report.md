# Technical Report: VNPT AI Hackathon - Parallel API Pipeline

##  Tổng quan dự án

Dự án xây dựng hệ thống inference cho cuộc thi VNPT AI Hackathon, sử dụng LLM API để trả lời câu hỏi trắc nghiệm đa lĩnh vực với độ chính xác cao và xử lý song song để tối ưu thời gian.

---

##  Kiến trúc hệ thống

### Thành phần chính

```
┌─────────────────────────────────────────────────────────────┐
│                        predict.py                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Pipeline  │──│ QuestionRouter│──│  VNPTAPIClient    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Answer Cache │  │   Prompts   │  │  Small/Large Model  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Files chính

| File | Mô tả |
|------|-------|
| `predict.py` | Pipeline chính - xử lý song song, cache, fallback |
| `question_router.py` | Phân loại câu hỏi và build prompt chuyên biệt |
| `vnpt_api_client.py` | Client gọi VNPT LLM API |
| `test_concurrent.py` | Test concurrent connection limits |
| `eval_v*.py` | Scripts đánh giá accuracy |

---

##  Parallel Processing System

### Concurrent Dual-Queue Architecture

```python
# Hai queue chạy ĐỒNG THỜI với ThreadPoolExecutor
pending_queues = {
    "small": len(small_questions),  # 5 workers
    "large": len(large_questions)   # 3 workers
}
```

### Queue-Aware Fallback Logic

```
┌──────────────────────────────────────────────────────────┐
│                    FALLBACK RULES                         │
├──────────────────────────────────────────────────────────┤
│  Large→Small: CHỈ khi pending_small == 0                 │
│  Small→Large: CHỈ khi pending_large == 0                 │
│                                                          │
│  Lý do: Tránh lãng phí quota chéo khi còn câu hỏi       │
└──────────────────────────────────────────────────────────┘
```

### Xử lý Rate Limit

```python
# Smart Queue Switching Flow
1. Small + Large chạy concurrent
2. Nếu bị rate limit → stop queue đó
3. Cập nhật pending_queues count
4. Chờ quota reset (rolling 60-min window)
5. Retry concurrent với pending questions
```

---

## Caching System

### Answer Cache Structure

```json
{
  "version": "v14",
  "count": 370,
  "answers": {
    "test_0001": {
      "question": "...",
      "choices": ["A", "B", "C", "D"],
      "extracted_answer": "A",
      "type": "reading",
      "model": "large",
      "model_responses": ["Full reasoning..."],
      "prompt_system": "You are...",
      "prompt_user": "Question...",
      "timestamp": "2025-12-13T22:00:00"
    }
  }
}
```

### Immediate Cache Persistence

```python
# Mỗi câu hoàn thành → lưu cache ngay
def _process_parallel_smart(...):
    for future in as_completed(future_to_q):
        if result["status"] == "success":
            self._save_answer_cache()  # Persist to disk immediately
```

**Anti-data-loss:** Cache được lưu ngay sau mỗi câu để tránh mất dữ liệu khi crash.

---

## Question Classification

### Loại câu hỏi

| Type | Model | Strategy |
|------|-------|----------|
| `MATH` | Large | 2-call verification |
| `PHYSICS` | Large | 2-call verification |
| `CHEMISTRY` | Large | 2-call verification |
| `READING` | Large | Multi-call consensus |
| `SAFETY` | Small | Single call với prompt đặc biệt |
| `GENERAL` | Small | Single call |

### Specialized Prompts

Mỗi loại câu hỏi có prompt chuyên biệt:
- **MATH/PHYSICS**: Chain-of-thought với verification
- **READING**: Trích dẫn từ văn bản
- **SAFETY**: Xử lý câu hỏi nhạy cảm

---

##  Answer Extraction

### Pattern Matching Strategy

```python
# Ưu tiên pattern "Đáp án cuối cùng"
patterns = [
    r'đáp án cuối cùng[:\s]*([A-J])',
    r'chọn đáp án[:\s]*([A-J])',
    # ... more patterns
]
```

### Validation

Accuracy achieved: **87.5%** trên test set với ground truth

---

##  Error Handling

### Rate Limit Handling

```python
class RateLimitError(Exception): pass
class LargeModelRateLimited(RateLimitError): pass

# Retry logic với exponential backoff
retry_delays = [2, 5]  # seconds
```

### Fallback Chain

```
1. Primary model fails
2. Check if target queue empty
3. If empty → fallback to other model
4. If both limited → wait for quota reset
5. If content filtered → select "cannot answer" option
```

---

## Performance Metrics

### Concurrent Limits (Tested)

| Model | Max Concurrent |
|-------|----------------|
| Small | 5 workers |
| Large | 3 workers |

### Processing Speed

- **Sequential**: ~3-5s per question
- **Parallel**: ~0.5-1s per question (với multiple workers)
- **Speedup**: ~5x improvement

---

## Configuration

### API Parameters

```python
{
    "temperature": 0.3,  # Lower for consistency
    "top_p": 0.85,
    "top_k": 10,
    "seed": 42,          # Reproducibility
    "max_tokens": 8192
}
```

### Worker Configuration

```python
SMALL_WORKERS = 5  # For general questions
LARGE_WORKERS = 3  # For complex questions (quota limited)
```

## Usage

### Basic Run

```bash
python predict.py --input test.json --output submission.csv --cache-version v14
```

### Resume from Cache

```bash
# Tự động resume từ cache version
python predict.py --input test.json --output submission.csv --cache-version v14
```

### Evaluation

```bash
python evaluate.py
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| Total Questions | 93 |
| STEM Accuracy | 92.0% |
| Reading Accuracy | 85.0% |
| Safety Accuracy | 71.4% |
| **Overall Accuracy** | **87.5%** |

