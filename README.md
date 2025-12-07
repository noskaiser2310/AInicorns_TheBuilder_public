# VNPT AI Hackathon - Track 2: The Builder

## Team: Just2Try

Vietnamese Multiple-Choice Question Answering System using VNPT AI LLM APIs.

---

## Mô tả

Hệ thống trả lời câu hỏi trắc nghiệm tiếng Việt sử dụng VNPT AI LLM APIs. Pipeline hỗ trợ 4 loại câu hỏi:
- **Reading Comprehension**: Đọc hiểu văn bản
- **Factual**: Kiến thức tổng hợp
- **Math/Logic**: Toán học và suy luận
- **Safety**: Câu hỏi về an toàn, pháp luật

---

## Kiến trúc

```
Input JSON ──> Question Router ──> VNPT LLM API ──> Answer Extract ──> submission.csv
              (classify type)    (Small/Large)    (regex parsing)
```

---

## Cấu trúc thư mục

```
├── predict.py           # Main inference pipeline
├── question_router.py   # Phân loại câu hỏi và routing
├── vnpt_api_client.py   # API client với rate limiting
├── evaluate.py          # Đánh giá trên validation set
├── test_api.py          # Test kết nối API
├── api-keys.json        # API credentials
├── requirements.txt     # Dependencies
└── data/
    ├── val.json         # Validation set (100 câu)
    └── test.json        # Test set (370 câu)
```

---

## Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/noskaiser2310/AInicorns_TheBuilder_public.git
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu hình API keys
Đặt file `api-keys.json` với format:
```json
[
  {"llmApiName": "LLM small", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."},
  {"llmApiName": "LLM large", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."},
  {"llmApiName": "LLM embedings", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."}
]
```

---

## Sử dụng

### Test API connection
```bash
python test_api.py
```

### Chạy inference trên test set
```bash
python predict.py --input data/test.json --output submission.csv
```

### Đánh giá trên validation set
```bash
python evaluate.py --val-path data/val.json --max 30 --output eval_30.json
```

---

## Rate Limits

| Model | Daily | Hourly |
|-------|-------|--------|
| Small | 1000 | 60 |
| Large | 500 | 40 |
| Embedding | 500 | 40 |

---

## Routing Strategy

| Question Type | Model | Lý do |
|---------------|-------|-------|
| Reading | Small | Context có sẵn |
| Factual | Small | Tiết kiệm quota |
| Safety | Small | Đơn giản |
| Math | Large | Cần suy luận |

---

## Kết quả Validation

| Metric | Value |
|--------|-------|
| Accuracy (10 câu) | 100% |
| Accuracy (30 câu) | 83% |
| Reading | 85% |
| Factual | 82% |
| Math | 83% |

---

## Output Format

File `submission.csv`:
```csv
qid,answer
test_0001,A
test_0002,B
...
```

---

## Team Members

- Team: Just2Try
- Track: 2 - The Builder
