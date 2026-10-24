"""
FastAPI backend for the AI Research Assistant.

Runs the existing research pipeline in a background thread so the HTTP
request returns immediately, and exposes live agent progress plus the final
report. Job state is kept in memory — this is a local, single-user tool.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import run_research_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

AGENTS = ("search", "reader", "writer", "critic")
MAX_JOBS = 25

app = FastAPI(title="AI Research Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_job(job_id: str, topic: str) -> dict:
    return {
        "id": job_id,
        "topic": topic,
        "status": "queued",
        "agents": {
            name: {"status": "pending", "detail": ""} for name in AGENTS
        },
        "report": None,
        "sources": [],
        "critic": None,
        "warnings": [],
        "revisions": 0,
        "error": None,
        "output_path": None,
        "created_at": _now(),
        "finished_at": None,
    }


def _snapshot(job: dict) -> dict:
    """Public, JSON-safe view of a job (extracted page content is omitted)."""
    return {
        "id": job["id"],
        "topic": job["topic"],
        "status": job["status"],
        "agents": {
            name: dict(state) for name, state in job["agents"].items()
        },
        "report": job["report"],
        "sources": list(job["sources"]),
        "critic": job["critic"],
        "warnings": list(job["warnings"]),
        "revisions": job["revisions"],
        "error": job["error"],
    }


def _serialize_source(source) -> dict:
    return {
        "index": source.index,
        "title": source.title,
        "url": source.url,
        "source_domain": source.source_domain,
        "published_date": source.published_date,
        "snippet": source.snippet,
        "scrape_success": source.scrape_success,
    }


def _prune_jobs_locked() -> None:
    """Drop the oldest finished jobs once the in-memory store is full."""
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        (job for job in _jobs.values() if job["status"] in ("completed", "failed")),
        key=lambda job: job["created_at"],
    )
    for job in finished:
        if len(_jobs) <= MAX_JOBS:
            break
        _jobs.pop(job["id"], None)


def _set_agent_status(job_id: str, event: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        agent = event.get("agent")
        if agent in job["agents"]:
            job["agents"][agent] = {
                "status": event.get("status", "pending"),
                "detail": event.get("detail", ""),
            }


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "1235" in message or "connection" in lowered or "connect" in lowered:
        return (
            "Could not reach LM Studio at http://localhost:1235/v1. "
            "Start LM Studio, load the qwen3.5-2b model, and try again. "
            f"(Details: {message})"
        )
    return message


def _fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for agent in job["agents"].values():
            if agent["status"] == "running":
                agent["status"] = "error"
                agent["detail"] = agent["detail"] or "Stopped after failure."
            elif agent["status"] == "pending":
                agent["status"] = "skipped"
                agent["detail"] = "Not run."
        job["status"] = "failed"
        job["error"] = error
        job["finished_at"] = _now()


def _run_job(job_id: str, topic: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "running"

    try:
        state = run_research_pipeline(
            topic,
            on_event=lambda event: _set_agent_status(job_id, event),
        )
    except Exception as exc:
        logger.exception("Research job %s failed", job_id)
        _fail_job(job_id, _friendly_error(exc))
        return

    report = state.report_draft.strip()
    if not report:
        _fail_job(
            job_id,
            "The pipeline finished without producing a report. "
            "Check the agent statuses and server logs for details.",
        )
        return

    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "completed"
        job["report"] = report
        job["sources"] = [_serialize_source(source) for source in state.sources]
        job["critic"] = (
            state.critic_feedback.model_dump()
            if state.critic_feedback is not None
            else None
        )
        job["warnings"] = list(state.warnings)
        job["revisions"] = state.revision_count
        job["output_path"] = state.output_path
        job["finished_at"] = _now()


@app.post("/api/research", status_code=202)
def start_research(request: ResearchRequest):
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = _new_job(job_id, topic)
        _prune_jobs_locked()
        snapshot = _snapshot(_jobs[job_id])

    threading.Thread(
        target=_run_job,
        args=(job_id, topic),
        daemon=True,
        name=f"research-{job_id[:8]}",
    ).start()

    return snapshot


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        return _snapshot(job)


@app.get("/api/jobs/{job_id}/report")
def download_report(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        status = job["status"]
        output_path = job["output_path"]

    if status != "completed" or not output_path:
        raise HTTPException(status_code=409, detail="The report is not ready yet.")

    path = Path(output_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved report file not found.")

    return FileResponse(
        path,
        media_type="text/markdown; charset=utf-8",
        filename=path.name,
    )
