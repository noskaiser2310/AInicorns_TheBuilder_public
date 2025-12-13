# 🏆 VNPT AI Hackathon - Track 2: The Builder

<div align="center">

![Team Just2Try](https://img.shields.io/badge/Team-Just2Try-blue?style=for-the-badge)
![VNPT AI](https://img.shields.io/badge/VNPT-AI%20Hackathon-orange?style=for-the-badge)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-76B900?style=for-the-badge&logo=nvidia)

**Vietnamese Multi-Domain Question Answering System**  
*Powered by VNPT AI LLM with Advanced Reasoning & Multi-Strategy Voting*

</div>

---

## 📋 Table of Contents
- [Pipeline Flow](#-pipeline-flow)
- [Data Processing](#-data-processing)
- [Resource Initialization](#-resource-initialization)
- [Project Structure](#-project-structure)
- [Docker Deployment](#-docker-deployment)
- [Team](#-team)

---

## 🔄 Pipeline Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌─────────────────┐    ┌─────────────────────────┐   │
│   │ private_test │───►│ Question Router │───►│   Strategy Selection    │   │
│   │    .json     │    │  (Classify &    │    │                         │   │
│   └──────────────┘    │   Route)        │    │  ┌─────────────────┐    │   │
│                       └─────────────────┘    │  │ READING → LARGE │    │   │
│                                              │  │ (2-call voting) │    │   │
│                                              │  ├─────────────────┤    │   │
│                                              │  │ MATH → LARGE    │    │   │
│                                              │  │ (Solve+Verify)  │    │   │
│                                              │  ├─────────────────┤    │   │
│                                              │  │ FACTUAL → SMALL │    │   │
│                                              │  │ (Single call)   │    │   │
│                                              │  ├─────────────────┤    │   │
│                                              │  │ SAFETY → SMALL  │    │   │
│                                              │  │ (Refusal)       │    │   │
│                                              │  └─────────────────┘    │   │
│                                              └─────────────────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                              ┌─────────────────────────┐   │
│                                              │   VNPT AI LLM API       │   │
│                                              │   (Small / Large)       │   │
│                                              └─────────────────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                              ┌─────────────────────────┐   │
│                                              │   Answer Extraction     │   │
│                                              │   (6-Level Priority)    │   │
│                                              └─────────────────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                              ┌─────────────────────────┐   │
│                                              │    submission.csv       │   │
│                                              │    (qid, answer)        │   │
│                                              └─────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Flow

#### Step 1: Question Classification (`question_router.py`)
```
Input Question → Analyze Content → Classify Type → Select Model → Build Prompt
```

| Type | Detection Method | Model | Strategy |
|------|------------------|-------|----------|
| **READING** | Contains passage + comprehension question | LARGE | 2-call voting |
| **MATH** | Contains numbers, equations, calculations | LARGE | Solve + Verify |
| **FACTUAL** | General knowledge (History, Law, Science) | SMALL | Single call |
| **SAFETY** | Harmful/sensitive content detection | SMALL | Refusal priority |

#### Step 2: LLM Processing (`predict.py`)
- **READING Questions**: 2 different prompts → 2 answers → Vote for majority
- **MATH Questions**: Solve → Verify solution → Final answer
- **FACTUAL Questions**: Domain-specific prompt → Single answer
- **SAFETY Questions**: Detect refusal option → Select safe answer

#### Step 3: Answer Extraction (6-Level Priority)
1. 🔴 `Đáp án cuối cùng: X` - Highest priority
2. 🟠 `**Đáp án: X**` - Bold pattern
3. 🟡 `Đáp án: X` - Standard pattern
4. 🟢 Last occurrence of answer pattern
5. 🔵 Standalone bold letter `**X**`
6. ⚪ Fallback to `A`

---

## 📊 Data Processing

### Input Format
```json
[
  {
    "qid": "test_0001",
    "question": "Câu hỏi tiếng Việt...",
    "choices": ["A. Đáp án 1", "B. Đáp án 2", "C. Đáp án 3", "D. Đáp án 4"]
  }
]
```

### Output Format
```csv
qid,answer
test_0001,A
test_0002,B
test_0003,C
```

### Data Flow
```
/code/private_test.json → predict.py → /code/submission.csv
```

### Question Categories Handled
| Category | Description | Strategy |
|----------|-------------|----------|
| Precision Critical | Safety/refusal questions | Prioritize "cannot answer" option |
| Compulsory | Must-answer correctly | High-accuracy prompts |
| RAG | Reading comprehension | Multi-approach voting |
| STEM | Math/Science | Step-by-step verification |
| Multidomain | General knowledge | Domain-specific prompts |

---

## ⚙️ Resource Initialization

### Prerequisites
- Python 3.8+ (Docker uses Python 3 from Ubuntu 20.04)
- VNPT API credentials (`api-keys.json`)

### API Keys Configuration
File `api-keys.json` should contain:
```json
[
  {"llmApiName": "LLM small", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."},
  {"llmApiName": "LLM large", "authorization": "Bearer ...", "tokenId": "...", "tokenKey": "..."}
]
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

**Required packages:**
- `requests>=2.28.0` - HTTP client for API calls
- `tqdm>=4.65.0` - Progress bar
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data manipulation

### No External Resources Required
This solution uses **VNPT AI LLM API only** - no additional:
- ❌ Vector Database
- ❌ Pre-trained model weights
- ❌ External indexing
- ❌ Local GPU inference

All processing is done via VNPT API calls.

---

## 📁 Project Structure

```
Just2Try_TheBuilder/
├── predict.py              # Main entry point - reads JSON, outputs CSV
├── question_router.py      # Question classification & prompt building
├── vnpt_api_client.py      # VNPT API client with rate limiting
├── inference.sh            # Docker entry point script
├── Dockerfile              # Container configuration (CUDA 12.2)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .dockerignore           # Exclude unnecessary files from build
```

### Core Files Description

| File | Purpose |
|------|---------|
| `predict.py` | Main pipeline: load questions → classify → call LLM → extract answer → save CSV |
| `question_router.py` | Classify question type, build appropriate prompts for each type |
| `vnpt_api_client.py` | Handle API calls with retry logic and rate limit handling |
| `inference.sh` | Entry point that runs `python predict.py` |

---

## 🐳 Docker Deployment

### Docker Hub Image
```
noskaiser231000/just2try_thebuilder:latest
```

### Build Locally
```bash
docker build -t just2try_thebuilder .
```

### Run Container
```bash
# BTC will run with:
docker run --gpus all \
  -v /path/to/api-keys.json:/code/api-keys.json \
  -v /path/to/private_test.json:/code/private_test.json \
  just2try_thebuilder
```

### Dockerfile Spec
- **Base Image**: `nvidia/cuda:12.2.0-devel-ubuntu20.04`
- **Entry Point**: `inference.sh`
- **Input**: `/code/private_test.json`
- **Output**: `/code/submission.csv`

### Submission Checklist
- [x] Dockerfile với CUDA 12.2 base
- [x] requirements.txt với tất cả dependencies
- [x] inference.sh entry point
- [x] Đọc `/code/private_test.json` → `/code/submission.csv`
- [x] Docker image pushed to Docker Hub

---

## 👥 Team

<div align="center">

### 🦄 Team Just2Try

**Track 2: The Builder**  
*VNPT AI Hackathon - Age of AInicorns 2024*

</div>

---

<div align="center">
Made with ❤️ by Team Just2Try
</div>
