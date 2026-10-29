"""A bounded, single-worker local-demo queue; pretrained model globals are serialized."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock

from app.services.pipeline_service import STAGES, run_pipeline


class JobStore:
    def __init__(self, runner=run_pipeline, repository=None):
        self.runner = runner
        self.repository = repository
        self.jobs = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verification")

    def create(self, job_id, image_path, filename, image_info, options):
        with self.lock:
            if sum(job["status"] in ("queued", "running") for job in self.jobs.values()) >= 3:
                raise ValueError("The analysis queue is full. Please wait for a current analysis to finish.")
            if len(self.jobs) >= 20:
                oldest = next((key for key, job in self.jobs.items() if job["status"] not in ("queued", "running")), None)
                if oldest:
                    self.jobs.pop(oldest)  # Uploaded files/reports are retained, not deleted.
            count = 10 if options.get("include_details") else 8
            self.jobs[job_id] = {
                "id": job_id, "filename": filename, "image": image_info,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued",
                "image_url": f"/api/analyses/{job_id}/image", "progress": 0,
                "stages": [{"key": key, "label": label, "state": "pending", "seconds": 0} for key, label in STAGES[:count]],
                "results": {}, "options": options,
            }
        self._persist(job_id)
        self.executor.submit(self._run, job_id, Path(image_path))
        return self.get(job_id)

    def _run(self, job_id, image_path):
        def emit(key, state, data, seconds):
            with self.lock:
                job = self.jobs[job_id]
                item = next(stage for stage in job["stages"] if stage["key"] == key)
                item.update(state=state, seconds=seconds)
                if data is not None:
                    job["results"][key] = data
                finished = sum(stage["state"] in ("completed", "error", "skipped") for stage in job["stages"])
                job["progress"] = round(finished / len(job["stages"]) * 100)
            self._persist(job_id)
        with self.lock:
            self.jobs[job_id]["status"] = "running"
            options = dict(self.jobs[job_id]["options"])
        self._persist(job_id)
        try:
            self.runner(str(image_path), options, emit)
            with self.lock:
                job = self.jobs[job_id]
                job["status"] = "completed_with_issues" if any(s["state"] == "error" for s in job["stages"]) else "completed"
                job["progress"] = 100
        except Exception as error:
            with self.lock:
                self.jobs[job_id].update(status="failed", error=str(error)[:1200])
        self._persist(job_id)
        snapshot = self.get(job_id)
        try:
            (image_path.parent / "report.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            # Results remain downloadable from memory if saving is unavailable.
            with self.lock:
                self.jobs[job_id]["save_warning"] = "Could not save the report to disk; download it from this session."

    def get(self, job_id):
        with self.lock:
            job = deepcopy(self.jobs.get(job_id))
        if job is not None or self.repository is None:
            return job
        try:
            return self.repository.get_analysis(job_id)
        except RuntimeError:
            return None

    def recent(self):
        with self.lock:
            memory = [{key: job[key] for key in ("id", "filename", "created_at", "status", "image_url")}
                      for job in reversed(list(self.jobs.values()))]
        if self.repository is None:
            return memory
        try:
            persisted = self.repository.recent_analyses(limit=20)
        except RuntimeError:
            return memory
        by_id = {job["id"]: job for job in persisted}
        by_id.update({job["id"]: job for job in memory})
        return sorted(by_id.values(), key=lambda job: job["created_at"], reverse=True)[:20]

    def _persist(self, job_id):
        if self.repository is None:
            return
        with self.lock:
            snapshot = deepcopy(self.jobs.get(job_id))
        if snapshot is None:
            return
        try:
            self.repository.save_analysis(snapshot)
        except RuntimeError:
            # MongoDB is optional; disk and in-memory behavior remain available.
            return

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
