# AI Resume Screening for Recruitment

FastAPI service that ranks resumes against a job description and recommends the best candidate using TF-IDF, top-K embeddings, and domain scoring.

## Features
- Upload and validate PDF resumes.
- Rank resumes by match percentage.
- Recommend a best candidate with domain scores and missing skills.
- Background jobs with progress tracking for long-running ranking.
- Two-stage ranking for speed: TF-IDF across all resumes, embeddings on top-K only.
- Embedding and resume text caching to reduce repeated compute.

## Requirements
- Python 3.9+
- Packages: `fastapi`, `uvicorn`, `pydantic`, `PyPDF2`, `scikit-learn`, `pandas`, `colorama`, `torch`, `transformers`

## Virtual Environment (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Run
1. Install dependencies:
```bash
pip install fastapi uvicorn pydantic PyPDF2 scikit-learn pandas colorama torch transformers
```
2. Start the API server:
```bash
uvicorn app:app --reload
```
3. Open the UI at `http://127.0.0.1:8000/`.
4. Upload PDF resumes and start a ranking or recommendation job.

## API Overview
### Upload resumes
- `POST /upload_resumes`

### Start ranking job
- `POST /rank`
- Body: `job_description`, `stream`, `top_n`
- Response: `{ "job_id": "..." }`

### Start recommendation job
- `POST /recommend`
- Body: `job_description`, `stream`
- Response: `{ "job_id": "..." }`

### Check job status
- `GET /job/{job_id}`
- Response: `progress`, `status`, and `result` when complete

### Additional endpoints
- `POST /analyze`
- `POST /analyze_candidate`
- `GET /streams`

## Notes
- Resumes are read from the `uploads/` directory.
- Job progress updates after each resume is processed.
- The embedding model downloads on first use; initial ranking may take longer.
- Embeddings use `BAAI/bge-small-en-v1.5` and are computed only for the top-K TF-IDF results.
- Fuzzy matching and evidence snippet extraction are limited to top-K resumes for speed.
- The UI polls job status about every 800-1000ms and stops immediately on `done` or `error`.






