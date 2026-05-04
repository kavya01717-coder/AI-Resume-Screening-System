# 🤖 AI Resume Screening System

An intelligent ATS (Applicant Tracking System) that automatically ranks and scores candidates from PDF resumes using semantic AI embeddings and multi-domain skill analysis.

---

## ✨ Features

- 🧠 **Semantic Matching** — Uses `BAAI/bge-small-en-v1.5` transformer embeddings with TF-IDF fallback
- 📊 **Multi-Domain Skill Scoring** — Scores resumes across 18+ domains (Engineering, Data Science, Finance, Sales, and more)
- 🏆 **Candidate Ranking** — Ranks all candidates by match percentage with configurable top-N results
- 🔍 **Mandatory Skill Filtering** — Must-have skills with pass/fail flags and gap detection
- ⚡ **Async Job Queue** — Background ranking with real-time progress polling
- 🛡️ **HR Override System** — Force-accept or force-reject candidates with notes
- 📝 **Audit Trails** — Every evaluation logs timestamps, stream, and job description used
- 🔒 **Secure Uploads** — Path traversal protection, 100 MB file size limit per PDF

---

## 🛠️ Tech Stack

`FastAPI` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `PyPDF2` · `pandas` · `Jinja2`

---

## 📁 Project Structure

```
ai-resume-screener/
├── app.py            # FastAPI routes and job queue
├── ats_engine.py     # Core ATS logic — embedding, ranking, scoring
├── templates/
│   └── index.html    # Frontend UI
├── uploads/          # Uploaded PDF resumes (auto-created)
└── requirements.txt
```

---

## 🚀 Getting Started

**1. Clone & install**
```bash
git clone https://github.com/your-username/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
```

**2. Run**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**3. Open** → `http://localhost:8000`

> On first run, the app downloads the `BAAI/bge-small-en-v1.5` model (~130 MB). Internet connection required.

---

## ⚙️ How It Works

1. Upload PDF resumes via the web UI
2. Enter a job description and select a stream (Engineering, Data Science, Accounting, etc.)
3. The engine embeds both the JD and each resume, computes cosine similarity, and applies keyword bonus scoring
4. Candidates are ranked by final score with skill gap analysis and a hire/no-hire recommendation

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_resumes` | Upload PDF resumes |
| `GET` | `/streams` | List available job streams |
| `POST` | `/recommend` | Start async ranking job, returns `job_id` |
| `GET` | `/job/{job_id}` | Poll job progress and result |
| `POST` | `/evaluate` | Synchronous full evaluation |
| `POST` | `/analyze_candidate` | Analyze a specific candidate |

---

## 📄 License

[MIT](LICENSE)
