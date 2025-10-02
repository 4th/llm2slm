import os, uuid, json, glob
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError
from typing import Dict, Any, List

from schemas import DistillConfig, JobStatus
from distiller import run_distillation

API_PREFIX = os.getenv("API_PREFIX", "")
DATA_DIR = os.getenv("DATA_DIR", "/data")
ART_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

api = FastAPI(title="LLM→SLM Distillation Service")

# Simple in-memory job registry (stateless alternative would be a store)
JOBS: Dict[str, JobStatus] = {}

@api.post(f"{API_PREFIX}/distill", response_model=JobStatus)
def submit_distill(cfg: DistillConfig):
    job_id = str(uuid.uuid4())[:8]
    job_path = os.path.join(ART_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    # Fire-and-forget within same process for demo; in prod use a queue/worker
    JOBS[job_id] = JobStatus(job_id=job_id, state="running", message="Started")
    try:
        run_distillation(cfg, job_path)
        artifacts = sorted([os.path.relpath(p, job_path) for p in glob.glob(os.path.join(job_path, "**/*"), recursive=True) if os.path.isfile(p)])
        JOBS[job_id] = JobStatus(job_id=job_id, state="completed", artifacts=artifacts, message="OK")
    except Exception as e:
        JOBS[job_id] = JobStatus(job_id=job_id, state="failed", message=str(e))
    return JOBS[job_id]

@api.get(f"{API_PREFIX}/status/{{job_id}}", response_model=JobStatus)
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job_id")
    return JOBS[job_id]

@api.get(f"{API_PREFIX}/artifacts/{{job_id}}")
def list_artifacts(job_id: str):
    job_path = os.path.join(ART_DIR, job_id)
    if not os.path.isdir(job_path):
        raise HTTPException(404, "Unknown job_id")
    files = sorted([os.path.relpath(p, job_path) for p in glob.glob(os.path.join(job_path, "**/*"), recursive=True) if os.path.isfile(p)])
    return {"job_id": job_id, "files": files}

@api.get(f"{API_PREFIX}/download/{{job_id}}/{{path:path}}")
def download(job_id: str, path: str):
    job_path = os.path.join(ART_DIR, job_id)
    abs_path = os.path.normpath(os.path.join(job_path, path))
    if not abs_path.startswith(job_path) or not os.path.isfile(abs_path):
        raise HTTPException(404, "File not found")
    return FileResponse(abs_path)
