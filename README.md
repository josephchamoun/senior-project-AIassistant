# EduGate AI Assistant

An AI-powered school assistant built with a full RAG (Retrieval Augmented Generation) pipeline, integrated into the EduGate school management system. Handles natural language questions from students, parents, and teachers — each with role-based access to their own data.

---

## How It Works

```
User sends message + Bearer token
            ↓
FastAPI validates identity with Laravel backend
            ↓
      Intent Classifier (3-layer hybrid system)
            ↓
Layer 1: Keyword matching          → fastest, no AI cost
Layer 2: Semantic similarity       → sentence-transformers embeddings
Layer 3: LLM classification        → fallback for ambiguous messages
            ↓
grades / attendance / schedule     → calls Laravel API (role-based, token-enforced)
policy / general question          → FAISS vector search over school policy documents
            ↓
Qwen2.5-3B-Instruct generates a human-like response using retrieved context
            ↓
Returns response + detected intent + functions called
```

---

## Key Technical Decisions

- **4-bit NF4 quantization** — compresses the 3B parameter model from ~12GB to ~2GB VRAM, making it run on a free Google Colab GPU
- **Hybrid intent classifier** — tries keyword matching first (0ms) before using AI similarity search, minimizing compute cost
- **FAISS IndexFlatL2** — exact nearest neighbor search to find the most relevant school policy for each question
- **Role-based access control** — students, parents, and teachers each see only their own data, enforced at the Laravel API level using Bearer tokens
- **FastAPI + ngrok** — exposes the Colab runtime as a public API endpoint for the frontend to call

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Qwen2.5-3B-Instruct (Hugging Face) |
| Quantization | BitsAndBytes (NF4 4-bit) |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Embeddings | Sentence Transformers |
| HTTP Client | httpx |
| Tunnel | pyngrok |
| Runtime | Google Colab (T4 GPU) |

---

## Role-Based Access

| Role | What they can ask |
|---|---|
| Student | Own grades, attendance, schedule, assignments, exams |
| Parent | Their children's grades, attendance, schedule |
| Teacher | Class schedules, materials — cannot access student grades |
| Admin | Full access to all school data |

---

## Related Repositories

- [EduGate Backend (Laravel)](https://github.com/MariaWadih/EduGate-Backend) — built by Joseph Chamoun & Maria Wadih
- [EduGate Frontend (React)](https://github.com/MariaWadih/EduGate-Frontend) — built by Joseph Chamoun & Maria Wadih

> **Team project:** The AI assistant module was designed and built entirely by **Joseph Chamoun**. The backend and frontend were co-developed by Joseph Chamoun and Maria Wadih.

---

## Setup

1. Open `chatbot_cleaner.py` in Google Colab and set runtime to GPU (T4)
2. Run all cells — dependencies install automatically
3. Enter your ngrok authtoken when prompted
4. Enter your Laravel backend base URL
5. Use the generated public ngrok URL as the AI endpoint in the frontend

---

## API Reference

**POST** `/chat`

Headers:
```
Authorization: Bearer <sanctum_token>
```

Body:
```json
{
  "message": "What are my grades this semester?",
}
```

Response:
```json
{
  "response": "Your grades this semester are...",
  "intent": "get_grades",
  "source": "laravel_api"
}
```
