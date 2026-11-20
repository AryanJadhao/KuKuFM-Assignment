"""One background thread per run, so the Streamlit page stays responsive while the LLM works.
Progress is read back from status.json, so a page refresh never loses it."""
from __future__ import annotations

import threading

from . import pipeline

_jobs: dict[str, tuple[threading.Thread, threading.Event]] = {}


def is_running(run_id: str) -> bool:
    j = _jobs.get(run_id)
    return bool(j and j[0].is_alive())


def start(run_id: str, settings=None) -> bool:
    """settings: optional Settings (e.g. with the model / key chosen in the app); default = .env."""
    if is_running(run_id):
        return False
    stop_event = threading.Event()

    def target():
        try:
            pipeline.run(run_id, settings=settings, stop=stop_event)
        except Exception:
            pass  # already recorded in status.json by pipeline.run

    t = threading.Thread(target=target, name=f"anchorcut-{run_id}", daemon=True)
    _jobs[run_id] = (t, stop_event)
    t.start()
    return True


def stop(run_id: str) -> None:
    j = _jobs.get(run_id)
    if j:
        j[1].set()
