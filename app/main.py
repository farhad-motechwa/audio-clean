"""Local web app: upload audio, clean it with a chosen engine, compare and download."""

import json
import re
import shutil
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import engines

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Rough processing time as a multiple of clip length, for the progress estimate.
SPEED = {"deepfilter": 0.15, "resemble-denoise": 0.4, "resemble-enhance": 4.0}

app = FastAPI(title="Audio Clean")
# One job at a time: the models are memory-hungry and share the CPU/GPU.
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^\w\-. ]+", "_", Path(name or "audio").name).strip(" .")
    return stem or "audio"


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _save(job: dict):
    (JOBS_DIR / job["id"] / "job.json").write_text(json.dumps(job, indent=2))


def _run(job_id: str):
    job = jobs[job_id]
    workdir = JOBS_DIR / job_id
    job.update(status="processing", started=time.time())
    _save(job)
    try:
        out = engines.process(workdir / job["input"], workdir, job["engine"], job["strength"],
                              job["loudnorm"], job["format"])
        job.update(status="done", output=out.name)
    except Exception as e:  # report any engine failure to the page
        traceback.print_exc()
        job.update(status="error", error=str(e) or e.__class__.__name__)
    job["finished"] = time.time()
    _save(job)


@app.get("/api/engines")
def list_engines():
    return [{"id": k, "label": v} for k, v in engines.ENGINES.items()]


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    engine: str = Form("deepfilter"),
    strength: float = Form(1.0),
    loudnorm: bool = Form(True),
    format: str = Form("wav"),
):
    if engine not in engines.ENGINES:
        raise HTTPException(400, f"Unknown engine: {engine}")
    if format not in ("wav", "mp3"):
        raise HTTPException(400, "Format must be wav or mp3")

    job_id = uuid.uuid4().hex[:12]
    workdir = JOBS_DIR / job_id
    workdir.mkdir(parents=True)
    name = _safe_name(file.filename)
    with (workdir / name).open("wb") as f:
        shutil.copyfileobj(file.file, f)

    duration = _duration(workdir / name)
    if duration <= 0:
        shutil.rmtree(workdir)
        raise HTTPException(400, "That file doesn't contain readable audio.")

    job = {
        "id": job_id, "input": name, "engine": engine, "strength": max(0.0, min(1.0, strength)),
        "loudnorm": loudnorm, "format": format, "duration": duration,
        "estimate": max(3.0, duration * SPEED[engine]), "status": "queued", "created": time.time(),
    }
    jobs[job_id] = job
    _save(job)
    executor.submit(_run, job_id)
    return job


def _job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job(job_id)
    ahead = sum(1 for j in jobs.values() if j["status"] in ("queued", "processing")
                and j["created"] < job["created"])
    return {**job, "ahead": ahead}


@app.get("/api/jobs/{job_id}/original")
def original(job_id: str):
    job = _job(job_id)
    return FileResponse(JOBS_DIR / job_id / job["input"])


@app.get("/api/jobs/{job_id}/result")
def result(job_id: str, download: bool = False):
    job = _job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "Not ready yet")
    path = JOBS_DIR / job_id / job["output"]
    return FileResponse(path, filename=path.name if download else None)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
