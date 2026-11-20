"""Live view of a run in progress, built only from files already on disk (no LLM calls).

Works for a run driven by the app's own thread or by a separate CLI process, because every stage
writes its output file as soon as a unit finishes:
  anchors/ep_XXX.json  -> an episode has been tagged
  critic/<id>.json     -> a candidate has been reviewed
  raw_outputs/*.json   -> a model call came back (one file per attempt)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import pipeline
from .models import Candidate, EpisodeScript, Promo
from .window_builder import build_candidates


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}   # file may be mid-write by the running process; it shows up on the next refresh


def signature(run_id: str) -> tuple:
    """Changes whenever any stage writes something - used as a cache key by the UI."""
    d = pipeline.run_dir(run_id)
    parts = []
    for sub in ("anchors", "critic", "raw_outputs"):
        files = list((d / sub).glob("*.json"))
        parts.append((len(files), max((f.stat().st_mtime for f in files), default=0)))
    for name in ("candidates.json", "status.json", "results/promos.json"):
        f = d / name
        parts.append(f.stat().st_mtime if f.exists() else 0)
    return tuple(parts)


def snapshot(run_id: str, episode_numbers: list[int]) -> dict:
    d = pipeline.run_dir(run_id)
    st = pipeline.status(run_id)
    now = time.time()
    events = []
    ep_state = {n: "pending" for n in episode_numbers}
    hooks = cliffs = dropped = 0
    anchor_times = []
    for f in (d / "anchors").glob("ep_*.json"):
        rec = _read(f)
        n = rec.get("episode")
        if n is None:
            continue
        acc = rec.get("accepted", [])
        h = sum(a["kind"] == "HOOK_ANCHOR" for a in acc)
        hooks += h
        cliffs += len(acc) - h
        dropped += len(rec.get("dropped", []))
        ep_state[n] = "done"
        mt = f.stat().st_mtime
        anchor_times.append(mt)
        events.append((mt, f"Episode {n}: {h} hook / {len(acc) - h} cliffhanger anchors"
                           + (f", {len(rec.get('dropped', []))} dropped" if rec.get("dropped") else "")))
    for n in st.get("anchors_failed", {}):
        if ep_state.get(int(n)) != "done":
            ep_state[int(n)] = "failed"
    live = st.get("state") == "running"
    if live and st.get("stage") == "anchors":
        flight = [n for n in st.get("in_flight", []) if isinstance(n, int)]
        if not flight and "in_flight" not in st:   # older runs without in-flight tracking: guess the next one
            flight = [n for n in episode_numbers if ep_state[n] == "pending"][:1]
        for n in flight:
            if ep_state.get(n) == "pending":
                ep_state[n] = "working"

    critic_files = list((d / "critic").glob("*.json"))
    for f in sorted(critic_files, key=lambda f: f.stat().st_mtime)[-6:]:
        v = _read(f).get("verdict") or {}
        if v:
            events.append((f.stat().st_mtime, f"Critic: {v.get('cand_id')} hook {v.get('first_3s_hook')}, "
                                              f"cliff {v.get('cliffhanger_open')}, clarity {v.get('cold_viewer_clarity')}"))
    raw = list((d / "raw_outputs").glob("*.json"))
    retries = sum(1 for f in raw if not f.stem.endswith("__1"))
    models = sorted({_read(f).get("model", "") for f in sorted(raw, key=lambda f: f.stat().st_mtime)[-5:]} - {""})

    n_cands = 0
    cand_rec = _read(d / "candidates.json") if (d / "candidates.json").exists() else {}
    if cand_rec:
        n_cands = len(cand_rec.get("candidates", []))
    done_eps = sum(s == "done" for s in ep_state.values())

    # ETA from the pace of finished episodes (anchors stage) or critic files (critic stage)
    eta_s = None
    stage = st.get("stage", "")
    if live and stage == "anchors" and len(anchor_times) >= 2:
        recent = sorted(anchor_times)[-9:]   # recent pace only: concurrency or a model switch changes it
        rate = (recent[-1] - recent[0]) / (len(recent) - 1)
        left_eps = len(episode_numbers) - done_eps
        eta_s = rate * left_eps + rate * max(1, round(len(episode_numbers) * 3 / 5))  # + critic, ~3 cands/ep, 5 per call
    elif live and stage == "critic" and n_cands:
        times = sorted(f.stat().st_mtime for f in critic_files)
        if len(times) >= 2:
            per_cand = (times[-1] - times[0]) / (len(times) - 1)
            eta_s = per_cand * (n_cands - len(critic_files))
    updated = st.get("updated_utc")
    age = None
    if updated:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
        except ValueError:
            pass
    events.sort(reverse=True)
    return {
        "state": st.get("state", "-"), "stage": stage, "progress": st.get("progress") or {}, "live": live,
        "episodes": ep_state, "episodes_done": done_eps, "episodes_total": len(episode_numbers),
        "hooks": hooks, "cliffs": cliffs, "dropped": dropped, "candidates": n_cands,
        "reviewed": len(critic_files), "model_calls": len(raw), "retries": retries, "models": models,
        "eta_s": eta_s, "status_age_s": age, "last_error": st.get("last_error"),
        "events": [(datetime.fromtimestamp(t).strftime("%H:%M:%S"), msg) for t, msg in events[:10]],
        "now": now,
    }


def provisional_promos(run_id: str, eps: list[EpisodeScript]) -> list[Promo]:
    """Promos as they stand right now.
    - after the run: the saved results
    - during the critic stage: saved windows + whatever verdicts exist (others show as Unreviewed)
    - during the anchor stage: windows built in memory from the episodes tagged so far (all Unreviewed)"""
    d = pipeline.run_dir(run_id)
    st = pipeline.status(run_id)
    if st.get("state") != "running" and (d / "results" / "promos.json").exists():
        return pipeline.load_promos(run_id)
    cands: list[Candidate] | None = None
    if not (d / "candidates.json").exists() or st.get("stage") == "anchors":
        anchors, _ = pipeline.load_anchors(run_id)
        if not anchors:
            return []
        s = pipeline.get_settings()
        cands, _ = build_candidates(eps, anchors, per_episode=s.candidates_per_episode)
    try:
        return pipeline.assemble(run_id, cands=cands, eps=eps)
    except (OSError, ValueError, KeyError):
        return pipeline.load_promos(run_id)
