import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import ats_engine
from ats_engine import analyze_best_candidate, rank_resumes

logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
STATIC_STREAMS = [
    "ENGINEERING",
    "DATA SCIENCE",
    "ACCOUNTING",
    "HR",
    "SALES",
    "MARKETING",
    "DESIGN",
]

jobs = {}
jobs_lock = threading.Lock()


@app.on_event("startup")
def load_embedding_on_startup() -> None:
    try:
        ats_engine.load_embedding_model()
        if (
            not getattr(ats_engine, "_embedding_loaded", False)
            or ats_engine.embedding_model is None
            or ats_engine.tokenizer is None
        ):
            logger.error("Embedding model failed to load on startup; using TF-IDF fallback.")
    except Exception:
        logger.exception("Embedding model failed to load on startup; using TF-IDF fallback.")


class OverrideRequest(BaseModel):
    force_accept: Optional[bool] = None
    force_reject: Optional[bool] = None
    note: Optional[str] = None


class RecommendRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    stream: str = Field(..., min_length=1)
    mandatory_skills: Optional[List[str]] = None
    override: Optional[OverrideRequest] = None


class AnalyzeCandidateRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=1)
    stream: str = Field(..., min_length=1)
    mandatory_skills: Optional[List[str]] = None
    override: Optional[OverrideRequest] = None


class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    stream: str = Field(..., min_length=1)
    top_n: Optional[int] = Field(5, ge=1, le=50)
    mandatory_skills: Optional[List[str]] = None


def resolve_candidate_path(resume_folder: str, filename: str) -> str:
    normalized = filename.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="filename must not be empty")

    safe_name = os.path.basename(normalized)
    if safe_name != normalized:
        raise HTTPException(status_code=400, detail="Invalid filename")

    candidate_path = os.path.abspath(os.path.join(resume_folder, safe_name))
    resume_root = os.path.abspath(resume_folder)
    if os.path.commonpath([candidate_path, resume_root]) != resume_root:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return candidate_path


def normalize_job_description(job_description: str, max_words: int = 100) -> str:
    tokens = re.findall(r"\S+", job_description)
    if len(tokens) > max_words:
        tokens = tokens[:max_words]
    return " ".join(tokens).strip()


def normalize_override(override: Optional[OverrideRequest]) -> Optional[Dict[str, Any]]:
    if not override:
        return None
    note = (override.note or "").strip()
    payload = {
        "force_accept": override.force_accept,
        "force_reject": override.force_reject,
        "note": note if note else None,
    }
    return payload


def build_audit_info(job_description: str, stream: str) -> Dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "timestamp": timestamp,
        "job_description_used": job_description,
        "stream_used": stream,
    }


def ensure_uploads_folder() -> str:
    uploads_dir = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir


def get_uploads_dir() -> str:
    return os.path.abspath(os.path.join(os.getcwd(), "uploads"))


def ensure_uploaded_pdfs() -> str:
    uploads_dir = get_uploads_dir()
    if not os.path.isdir(uploads_dir):
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    pdf_files = [
        name for name in os.listdir(uploads_dir)
        if os.path.isfile(os.path.join(uploads_dir, name)) and name.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    return uploads_dir


def create_job() -> str:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "progress": 0,
            "status": "running",
        }
    return job_id


def update_job(job_id: str, progress: Optional[int] = None, status: Optional[str] = None, result=None) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if progress is not None:
            job["progress"] = progress
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result


def run_rank_job(job_id: str, payload: RankRequest) -> None:
    start_time = time.perf_counter()
    try:
        job_description = normalize_job_description(payload.job_description.strip())
        if not job_description:
            raise HTTPException(status_code=400, detail="job_description must not be empty")

        resume_folder = ensure_uploaded_pdfs()

        def progress_callback(percent: int) -> None:
            update_job(job_id, progress=percent)

        sorted_results = rank_resumes(
            job_description,
            resume_folder=resume_folder,
            use_embedding=True,
            progress_callback=progress_callback,
            stream=payload.stream,
            mandatory_skills=payload.mandatory_skills,
            return_details=True,
        )
        if not sorted_results:
            raise HTTPException(status_code=404, detail="No resumes found in selected stream.")

        logger.info(
            "Rank job %s completed in %.2fs with %d results",
            job_id,
            time.perf_counter() - start_time,
            len(sorted_results),
        )

        ranking = [
            {
                "rank": index + 1,
                "filename": candidate["filename"],
                "match_percentage": round(candidate["match_percentage"], 2),
                "remark": generate_remark(candidate["match_percentage"]),
                "must_have_pass": candidate.get("must_have_pass"),
                "missing_mandatory_skills": candidate.get("missing_mandatory_skills", []),
                "confidence_level": candidate.get("confidence_level"),
                "evidence_map": candidate.get("evidence_map", {}),
                "score_breakdown": candidate.get("score_breakdown", {}),
            }
            for index, candidate in enumerate(sorted_results[:payload.top_n])
        ]

        audit_info = build_audit_info(job_description, payload.stream)
        update_job(job_id, progress=100, status="done", result={"ranking": ranking, "audit_info": audit_info})

    except HTTPException as exc:
        logger.warning(
            "Rank job %s failed after %.2fs: %s",
            job_id,
            time.perf_counter() - start_time,
            exc.detail,
        )
        update_job(job_id, progress=100, status="error", result={"error": exc.detail})
    except Exception as exc:
        logger.exception(
            "Rank job %s crashed after %.2fs",
            job_id,
            time.perf_counter() - start_time,
        )
        update_job(job_id, progress=100, status="error", result={"error": str(exc)})


def run_recommend_job(job_id: str, payload: RecommendRequest) -> None:
    start_time = time.perf_counter()
    try:
        job_description = normalize_job_description(payload.job_description.strip())
        if not job_description:
            raise HTTPException(status_code=400, detail="job_description must not be empty")

        resume_folder = ensure_uploaded_pdfs()

        def progress_callback(percent: int) -> None:
            update_job(job_id, progress=percent)

        sorted_results = rank_resumes(
            job_description,
            resume_folder=resume_folder,
            use_embedding=True,
            progress_callback=progress_callback,
            stream=payload.stream,
            mandatory_skills=payload.mandatory_skills,
            return_details=True,
        )
        if not sorted_results:
            raise HTTPException(status_code=404, detail="No resumes found in selected stream.")

        best_candidate, scored, total_score, recommendation = analyze_best_candidate(
            sorted_results,
            resume_folder=resume_folder,
            override=normalize_override(payload.override),
            progress_callback=progress_callback,
        )
        score_column = "   score earned"
        if score_column not in scored.columns:
            raise HTTPException(status_code=500, detail="Expected score column not found in analysis")

        resume_path = os.path.join(resume_folder, best_candidate)
        resume_content = ats_engine.extract_and_clean_resume(resume_path)
        job_tokens = re.findall(r"[a-zA-Z0-9]+", job_description.lower())
        job_keywords = [token for token in job_tokens if len(token) > 2]
        resume_tokens = set(re.findall(r"[a-zA-Z0-9]+", resume_content))
        missing_skills = []
        for token in job_keywords:
            if token not in resume_tokens and token not in missing_skills:
                missing_skills.append(token)
            if len(missing_skills) >= 10:
                break

        ranking = [
            {
                "rank": index + 1,
                "filename": candidate["filename"],
                "match_percentage": round(candidate["match_percentage"], 2),
                "remark": generate_remark(candidate["match_percentage"]),
                "must_have_pass": candidate.get("must_have_pass"),
                "missing_mandatory_skills": candidate.get("missing_mandatory_skills", []),
                "confidence_level": candidate.get("confidence_level"),
                "evidence_map": candidate.get("evidence_map", {}),
                "score_breakdown": candidate.get("score_breakdown", {}),
            }
            for index, candidate in enumerate(sorted_results[:5])
        ]
        skills = [
            {
                "domain": domain,
                "score": score,
            }
            for domain, score in scored[score_column].to_dict().items()
        ]

        best_detail = sorted_results[0]
        audit_info = build_audit_info(job_description, payload.stream)
        result = {
            "best_candidate": best_candidate,
            "total_score": total_score,
            "recommendation": recommendation,
            "remark": generate_remark(sorted_results[0]["match_percentage"]),
            "ranking": ranking,
            "skills": skills,
            "missing_skills": missing_skills,
            "must_have_pass": best_detail.get("must_have_pass"),
            "missing_mandatory_skills": best_detail.get("missing_mandatory_skills", []),
            "confidence_level": best_detail.get("confidence_level"),
            "evidence_map": best_detail.get("evidence_map", {}),
            "score_breakdown": best_detail.get("score_breakdown", {}),
            "override": normalize_override(payload.override),
            "audit_info": audit_info,
        }

        update_job(job_id, progress=100, status="done", result=result)

        logger.info(
            "Recommend job %s completed in %.2fs",
            job_id,
            time.perf_counter() - start_time,
        )

    except HTTPException as exc:
        logger.warning(
            "Recommend job %s failed after %.2fs: %s",
            job_id,
            time.perf_counter() - start_time,
            exc.detail,
        )
        update_job(job_id, progress=100, status="error", result={"error": exc.detail})
    except Exception as exc:
        logger.exception(
            "Recommend job %s crashed after %.2fs",
            job_id,
            time.perf_counter() - start_time,
        )
        update_job(job_id, progress=100, status="error", result={"error": str(exc)})


def generate_remark(score: float) -> str:
    if score >= 85:
        return "Excellent alignment with role requirements"
    if score >= 70:
        return "Strong profile with high relevance"
    if score >= 55:
        return "Good match with some gaps"
    if score >= 40:
        return "Partial alignment, may require review"
    if score >= 25:
        return "Limited match to key skills"
    return "Low alignment with current needs"


def validate_pdf_upload(upload: UploadFile) -> None:
    filename = (upload.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    safe_name = os.path.basename(filename)
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content_type = (upload.content_type or "").lower()
    if content_type and content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")


def save_upload_with_limit(upload: UploadFile, target_path: str, max_bytes: int) -> None:
    bytes_written = 0
    try:
        with open(target_path, "wb") as output_file:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds 100MB limit")
                output_file.write(chunk)
    except Exception:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/rank")
def rank_system():
    return {
        "message": "Use POST /rank with job description and stream instead."
    }


@app.get("/analyze")
def analyze_system():
    return {
        "message": "Use POST /analyze with job description and stream instead."
    }


@app.get("/streams")
def list_streams():
    return {
        "streams": STATIC_STREAMS
    }


@app.post("/rank")
def rank_system_post(payload: RankRequest):
    job_id = create_job()
    thread = threading.Thread(target=run_rank_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/analyze")
def analyze_system_post(payload: RecommendRequest):
    try:
        job_description = normalize_job_description(payload.job_description.strip())
        stream = payload.stream

        if not job_description:
            raise HTTPException(status_code=400, detail="job_description must not be empty")

        resume_folder = ensure_uploaded_pdfs()
        sorted_results = rank_resumes(
            job_description,
            resume_folder=resume_folder,
            use_embedding=True,
            stream=payload.stream,
            mandatory_skills=payload.mandatory_skills,
            return_details=True,
        )
        if not sorted_results:
            raise HTTPException(status_code=404, detail="No resumes found in selected stream.")

        best_candidate, scored, total_score, recommendation = analyze_best_candidate(
            sorted_results,
            resume_folder=resume_folder,
            override=normalize_override(payload.override),
        )
        score_column = "   score earned"
        if score_column not in scored.columns:
            raise HTTPException(status_code=500, detail="Expected score column not found in analysis")

        best_detail = sorted_results[0]
        audit_info = build_audit_info(job_description, payload.stream)
        return {
            "best_candidate": best_candidate,
            "total_score": total_score,
            "recommendation": recommendation,
            "must_have_pass": best_detail.get("must_have_pass"),
            "missing_mandatory_skills": best_detail.get("missing_mandatory_skills", []),
            "confidence_level": best_detail.get("confidence_level"),
            "evidence_map": best_detail.get("evidence_map", {}),
            "score_breakdown": best_detail.get("score_breakdown", {}),
            "override": normalize_override(payload.override),
            "audit_info": audit_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend_system(payload: RecommendRequest):
    job_id = create_job()
    thread = threading.Thread(target=run_recommend_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/evaluate")
def evaluate_system(payload: RecommendRequest):
    try:
        job_description = normalize_job_description(payload.job_description.strip())
        if not job_description:
            raise HTTPException(status_code=400, detail="job_description must not be empty")

        resume_folder = ensure_uploaded_pdfs()

        sorted_results = rank_resumes(
            job_description,
            resume_folder=resume_folder,
            use_embedding=True,
            stream=payload.stream,
            mandatory_skills=payload.mandatory_skills,
            return_details=True,
        )
        if not sorted_results:
            raise HTTPException(status_code=404, detail="No resumes found in selected stream.")

        best_candidate, scored, total_score, recommendation = analyze_best_candidate(
            sorted_results,
            resume_folder=resume_folder,
            override=normalize_override(payload.override),
        )
        score_column = "   score earned"
        if score_column not in scored.columns:
            raise HTTPException(status_code=500, detail="Expected score column not found in analysis")

        resume_path = os.path.join(resume_folder, best_candidate)
        resume_content = ats_engine.extract_and_clean_resume(resume_path)
        job_tokens = re.findall(r"[a-zA-Z0-9]+", job_description.lower())
        job_keywords = [token for token in job_tokens if len(token) > 2]
        resume_tokens = set(re.findall(r"[a-zA-Z0-9]+", resume_content))
        missing_skills = []
        for token in job_keywords:
            if token not in resume_tokens and token not in missing_skills:
                missing_skills.append(token)
            if len(missing_skills) >= 10:
                break

        ranking = [
            {
                "rank": index + 1,
                "filename": candidate["filename"],
                "match_percentage": round(candidate["match_percentage"], 2),
                "remark": generate_remark(candidate["match_percentage"]),
                "must_have_pass": candidate.get("must_have_pass"),
                "missing_mandatory_skills": candidate.get("missing_mandatory_skills", []),
                "confidence_level": candidate.get("confidence_level"),
                "evidence_map": candidate.get("evidence_map", {}),
                "score_breakdown": candidate.get("score_breakdown", {}),
            }
            for index, candidate in enumerate(sorted_results)
        ]
        skills = [
            {
                "domain": domain,
                "score": score,
            }
            for domain, score in scored[score_column].to_dict().items()
        ]

        best_detail = sorted_results[0]
        audit_info = build_audit_info(job_description, payload.stream)
        return {
            "ranking": ranking,
            "best_candidate": best_candidate,
            "total_score": total_score,
            "recommendation": recommendation,
            "skills": skills,
            "missing_skills": missing_skills,
            "must_have_pass": best_detail.get("must_have_pass"),
            "missing_mandatory_skills": best_detail.get("missing_mandatory_skills", []),
            "confidence_level": best_detail.get("confidence_level"),
            "evidence_map": best_detail.get("evidence_map", {}),
            "score_breakdown": best_detail.get("score_breakdown", {}),
            "override": normalize_override(payload.override),
            "audit_info": audit_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "progress": job["progress"],
        "status": job["status"],
    }
    if job["status"] in {"done", "error"}:
        response["result"] = job.get("result")
    return response


@app.post("/analyze_candidate")
def analyze_candidate(payload: AnalyzeCandidateRequest):
    try:
        filename = payload.filename.strip()
        job_description = normalize_job_description(payload.job_description.strip())
        stream = payload.stream

        if not filename:
            raise HTTPException(status_code=400, detail="filename must not be empty")
        if not job_description:
            raise HTTPException(status_code=400, detail="job_description must not be empty")

        resume_folder = ensure_uploaded_pdfs()

        candidate_path = resolve_candidate_path(resume_folder, filename)
        if not os.path.isfile(candidate_path):
            raise HTTPException(status_code=404, detail="Candidate resume not found")

        sorted_results = rank_resumes(
            job_description,
            resume_folder=resume_folder,
            use_embedding=True,
            stream=payload.stream,
            mandatory_skills=payload.mandatory_skills,
            return_details=True,
        )
        if not sorted_results:
            raise HTTPException(status_code=404, detail="No resumes found for this stream")

        forced_sorted_results = [
            item for item in sorted_results if item.get("filename") == filename
        ]
        if not forced_sorted_results:
            forced_sorted_results = [{
                "filename": filename,
                "match_percentage": 0.0,
                "must_have_pass": False,
                "missing_mandatory_skills": payload.mandatory_skills or [],
                "confidence_level": "Low",
                "evidence_map": {},
                "score_breakdown": {
                    "similarity_score": 0.0,
                    "bonus_score": 0.0,
                    "final_score": 0.0,
                },
            }]
        forced_sorted_results.extend(
            [item for item in sorted_results if item.get("filename") != filename]
        )

        best_candidate, scored, total_score, recommendation = analyze_best_candidate(
            forced_sorted_results,
            resume_folder=resume_folder,
            override=normalize_override(payload.override),
        )

        resume_content = ats_engine.extract_and_clean_resume(candidate_path)
        job_tokens = re.findall(r"[a-zA-Z0-9]+", job_description.lower())
        job_keywords = [token for token in job_tokens if len(token) > 2]
        resume_tokens = set(re.findall(r"[a-zA-Z0-9]+", resume_content))
        missing_skills = []
        for token in job_keywords:
            if token not in resume_tokens and token not in missing_skills:
                missing_skills.append(token)
            if len(missing_skills) >= 10:
                break

        best_detail = forced_sorted_results[0]
        audit_info = build_audit_info(job_description, payload.stream)
        return {
            "best_candidate": best_candidate,
            "total_score": total_score,
            "recommendation": recommendation,
            "missing_skills": missing_skills,
            "must_have_pass": best_detail.get("must_have_pass"),
            "missing_mandatory_skills": best_detail.get("missing_mandatory_skills", []),
            "confidence_level": best_detail.get("confidence_level"),
            "evidence_map": best_detail.get("evidence_map", {}),
            "score_breakdown": best_detail.get("score_breakdown", {}),
            "override": normalize_override(payload.override),
            "audit_info": audit_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_resumes")
def upload_resumes(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        uploads_dir = ensure_uploads_folder()
        uploaded_files = []

        for upload in files:
            validate_pdf_upload(upload)

            safe_name = os.path.basename(upload.filename)
            target_path = os.path.abspath(os.path.join(uploads_dir, safe_name))
            if os.path.commonpath([target_path, uploads_dir]) != uploads_dir:
                raise HTTPException(status_code=400, detail="Invalid filename")

            save_upload_with_limit(upload, target_path, MAX_UPLOAD_BYTES)
            uploaded_files.append(safe_name)

        return {"uploaded": uploaded_files}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
