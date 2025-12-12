# 🏆 VNPT AI Hackathon - Track 2: The Builder

<div align="center">

![Team Just2Try](https://img.shields.io/badge/Team-Just2Try-blue?style=for-the-badge)
![VNPT AI](https://img.shields.io/badge/VNPT-AI%20Hackathon-orange?style=for-the-badge)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)

**Vietnamese Multi-Domain Question Answering System**  
*Powered by VNPT AI LLM with Advanced Reasoning & Multi-Strategy Voting*

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Question Types & Strategies](#-question-types--strategies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Docker Deployment](#-docker-deployment)
- [Team](#-team)

---

## 🎯 Overview

A high-accuracy Vietnamese question answering system designed for the **VNPT AI Hackathon - Age of Just2Try**. The system handles 5 domain categories:

| Domain | Description |
|--------|-------------|
| **Precision Critical** | Questions requiring refusal/safety responses |
| **Compulsory** | Must-answer questions with high accuracy |
| **RAG** | Long-form reading comprehension |
| **STEM** | Mathematics and logical reasoning |
| **Multidomain** | General knowledge across fields |

---

## ✨ Key Features

### 🧠 Intelligent Question Routing
- Automatic classification into READING, MATH, FACTUAL, SAFETY types
- Sub-type detection (History, Law, Geography, Science, etc.)
- Dynamic model selection (Small vs Large) based on complexity

### 🗳️ Multi-Strategy Voting System
- **3-Approach Voting** for READING: Quote-Match, Elimination, Summary
- **2-Step Verification** for MATH: Solve → Verify → Confirm
- Majority voting with conflict resolution

### 🔍 Robust Answer Extraction
- 6-priority extraction system with "Đáp án cuối cùng" priority
- Bold pattern detection (**A**)
- Fallback mechanisms for edge cases

### ⚡ Smart Rate Limiting
- Rolling 60-minute window detection
- Automatic wait and retry with quota reset
- Graceful fallback between models

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input (JSON) ──► Question Router ──► Strategy Selection                │
│                   │                                                      │
│                   ├── READING ────► LARGE ──► 3-Approach Voting         │
│                   │                          (Quote / Eliminate / Sum)   │
│                   │                                                      │
│                   ├── MATH ───────► LARGE ──► Solve + Verify (2 calls)  │
│                   │                                                      │
│                   ├── FACTUAL ────► SMALL ──► Single Call + Analysis    │
│                   │                                                      │
│                   └── SAFETY ─────► SMALL ──► Single Call (Refusal)     │
│                                                                          │
│                              ▼                                           │
│                    Answer Extraction (6-Level Priority)                  │
│                              ▼                                           │
│                       submission.csv                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Question Types & Strategies

| Type | Model | Strategy | API Calls | Description |
|------|-------|----------|-----------|-------------|
| **READING** | LARGE | 3-Approach Voting | 3 | Quote-Match, Elimination, Summary methods |
| **MATH** | LARGE | 2-Step Verification | 2 | Solve → Examiner Verify |
| **FACTUAL** | SMALL | Analysis Method | 1 | Domain-specific prompts (Law, History, etc.) |
| **SAFETY** | SMALL | Direct Response | 1 | Prioritize refusal options |

### Answer Extraction Priority
1. 🔴 `Đáp án cuối cùng: X` - Highest priority
2. 🟠 `✅ Đáp án: X` or `**Đáp án: X**`
3. 🟡 Standard patterns: `Đáp án: X`, `Kết luận...Đáp án: X`
4. 🟢 Last match of `Đáp án: X` in text
5. 🔵 Standalone bold `**X**` at end
6. ⚪ Fallback to A

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- VNPT API credentials

### Setup
```bash
# Clone repository
git clone https://github.com/your-repo/Just2Try_TheBuilder.git
cd Just2Try_TheBuilder

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp api-keys.example.json api-keys.json
# Edit api-keys.json with your credentials
```

### API Keys Format
```json
[
  {"llmApiName": "LLM small", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."},
  {"llmApiName": "LLM large", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."},
  {"llmApiName": "LLM embedings", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."}
]
```

---

## 💻 Usage

### Inference
```bash
# Run on test set
python predict.py --input data/private_test.json --output submission.csv

# With custom cache version
python predict.py --input data/test.json --output submission.csv --cache-version v3
```

### Evaluation
```bash
# Evaluate questions 1-50
python evaluate.py --start 1 --end 50

# Full validation set
python evaluate.py --start 1 --end 93
```

### Build Legal RAG Index (Optional)
```bash
# Build BM25 index from legal corpus
python legal_rag_builder.py --json data/datasets/legal_corpus/legal_corpus.json

# Evaluate RAG quality
python legal_rag_eval.py --questions 20
```

---

## 📁 Project Structure

```
Just2Try_TheBuilder/
├── 📄 Core Files
│   ├── predict.py              # Main inference pipeline
│   ├── question_router.py      # Question classification & prompt building
│   ├── vnpt_api_client.py      # API client with rate limiting
│   └── evaluate.py             # Evaluation on validation set
│
├── 📄 RAG System (Optional)
│   ├── legal_rag_builder.py    # Build legal corpus index
│   ├── legal_rag.py            # Hybrid search (BM25 + Semantic)
│   └── legal_rag_eval.py       # RAG evaluation
│
├── 📄 Docker
│   ├── Dockerfile              # Container configuration
│   ├── inference.sh            # Entry point script
│   └── requirements.txt        # Python dependencies
│
├── 📄 Data
│   ├── data/val.json           # Validation set
│   ├── data/test.json          # Test set
│   └── data/datasets/          # Legal corpus datasets
│
└── 📄 Config
    ├── api-keys.json           # API credentials (gitignored)
    └── .dockerignore           # Docker ignore rules
```

---

## 🔧 Technical Details

### Rate Limiting
- **Small Model**: 60 req/hour, 1000 req/day
- **Large Model**: 40 req/hour, 500 req/day
- **Embedding**: 500 req/minute

### Retry Strategy
| Error Type | Action |
|------------|--------|
| Rate Limit (429) | Exponential backoff (5s → 80s) |
| Server Error | Wait 60s → 120s → Switch model |
| Both Models Fail | Wait 65 minutes (rolling window) |

### Caching
- Answers cached by question ID + cache version
- Resume capability for interrupted runs
- Cache stored in `answer_cache_v{version}.json`

---

## 🐳 Docker Deployment

### Build
```bash
docker build -t Just2Try_thebuilder .
```

### Run
```bash
# With GPU support
docker run --gpus all -v /path/to/data:/code Just2Try_thebuilder

# CPU only
docker run -v /path/to/data:/code Just2Try_thebuilder
```

### Submission Checklist
- [x] Dockerfile với CUDA 12.2 base
- [x] requirements.txt với tất cả dependencies
- [x] inference.sh entry point
- [x] Đọc `/code/private_test.json` → `/code/submission.csv`
- [x] Team name: Just2Try

---

## 👥 Team

<div align="center">

### 🦄 Team Just2Try

**Track 2: The Builder**  
*VNPT AI Hackathon - Age of Just2Try 2024*

</div>

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 80%+ |
| READING Accuracy | 80%+ |
| MATH Accuracy | 72%+ |
| FACTUAL Accuracy | 82%+ |

*Note: Results may vary based on API response quality and rate limits.*

---

## 📜 License

This project is developed for the VNPT AI Hackathon competition.

---

<div align="center">
Made with ❤️ by Team Just2Try
</div>
