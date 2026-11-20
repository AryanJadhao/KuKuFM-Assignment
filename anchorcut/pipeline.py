"""Run orchestration. Every stage writes to disk; re-running skips finished work (resume).

workspace/runs/<run_id>/
  status.json                       stage, progress, errors (drives the UI and resume)
  parsed/episodes.json              deterministic parse        parsed/parse_issues.json
  raw_outputs/*.json                EVERY LLM call, unedited: prompt + raw response + model + time
  anchors/ep_XXX.json               model output for the episode + accepted / dropped anchors
  candidates.json                   windows built by code (rebuilt from anchors each time)
  critic/<cand_id>.json             critic verdict per candidate
  results/promos.json  results/posting_schedule.csv  results/quality_report.md

The uploaded ZIP is not stored; only the parsed beats are (the pack is confidential).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
import traceback
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .anchor_tagger import tag_episode
from .cold_viewer_critic import apply_verdict, clip_lines, critique_batch, schedule
from .config import Settings, get_settings
from .llm_client import LLMClient, LLMError, QuotaPause
from .models import TIER_LABEL, Anchor, Candidate, CriticVerdictLLM, EpisodeScript, ParseIssue, Promo
from .parser import fmt, load_pack
from .verify import verify
from .window_builder import Index, build_candidates

log = logging.getLogger("anchorcut")
Progress = Callable[[str, int, int, str], None]
_lock = threading.Lock()


# ------------------------------------------------------------------ storage helpers
def runs_root(s: Settings | None = None) -> Path:
    p = (s or get_settings()).workspace / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(run_id: str) -> Path:
    d = runs_root() / run_id
    if not d.resolve().is_relative_to(runs_root().resolve()):
        raise ValueError("bad run id")
    return d


def list_runs() -> list[str]:
    return sorted((p.name for p in runs_root().iterdir() if (p / "status.json").exists()), reverse=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return default


def status(run_id: str) -> dict:
    return read_json(run_dir(run_id) / "status.json", {}) or {}


def set_status(run_id: str, **kw) -> dict:
    with _lock:
        st = status(run_id)
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(st.get(k), dict):
                st[k].update(v)
            else:
                st[k] = v
        st["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(run_dir(run_id) / "status.json", st)
        return st


# ------------------------------------------------------------------ stage 0: ingest
def pack_hash(zip_bytes: bytes) -> str:
    return hashlib.sha256(zip_bytes).hexdigest()


def find_run_for_pack(zip_bytes: bytes) -> str | None:
    """An existing run of this exact ZIP, preferring the one with the most work done."""
    h = pack_hash(zip_bytes)
    same = [r for r in list_runs() if status(r).get("pack_sha256") == h]
    return max(same, key=lambda r: (len(status(r).get("anchors_done", [])), r), default=None)


def ingest(zip_bytes: bytes, label: str = "") -> str:
    episodes, pack_issues = load_pack(zip_bytes)
    if not episodes:
        raise ValueError("no .txt screenplays found in the ZIP")
    if not any(e.beats for e in episodes):
        raise ValueError("no timestamped beats could be parsed from any screenplay")
    safe = "".join(c if c.isalnum() else "-" for c in label.lower())[:24].strip("-")
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{safe}" if safe else "")
    run_id, n = base, 1
    while run_dir(run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    d = run_dir(run_id)
    write_json(d / "parsed" / "episodes.json", [e.model_dump() for e in episodes])
    issues = [i.model_dump() for i in pack_issues] + [{**i.model_dump(), "episode": e.episode} for e in episodes for i in e.issues]
    write_json(d / "parsed" / "parse_issues.json", issues)
    set_status(run_id, id=run_id, label=label, state="parsed", stage="parsed", pack_sha256=pack_hash(zip_bytes),
               episodes=[e.episode for e in episodes], beats=sum(len(e.beats) for e in episodes),
               parse_issue_count=len(issues), anchors_done=[], anchors_failed={}, critic_failed={},
               created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return run_id


def load_episodes(run_id: str) -> list[EpisodeScript]:
    return [EpisodeScript.model_validate(e) for e in read_json(run_dir(run_id) / "parsed" / "episodes.json", [])]


# ------------------------------------------------------------------ parallel helper
def _parallel(run_id: str, items: list, worker: Callable, n: int, stop: threading.Event | None,
              on_done: Callable[[int], None], label: Callable[[object], object]) -> None:
    """Run worker(item) on up to n threads. Units are independent and each writes its own output file,
    so order does not matter and a crash loses at most the calls in flight. A QuotaPause in any worker
    stops new work from starting; calls already in flight finish, then the pause is raised."""
    pause = threading.Event()
    paused: list[QuotaPause] = []
    flight: set = set()

    def guarded(item):
        if pause.is_set() or (stop is not None and stop.is_set()):
            return
        with _lock:
            flight.add(label(item))
        set_status(run_id, in_flight=sorted(flight, key=str))
        try:
            worker(item)
        except QuotaPause as e:
            pause.set()
            paused.append(e)
        finally:
            with _lock:
                flight.discard(label(item))
            set_status(run_id, in_flight=sorted(flight, key=str))

    with ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="anchorcut") as ex:
        futures = [ex.submit(guarded, it) for it in items]
        for k, f in enumerate(as_completed(futures), 1):
            f.result()
            on_done(k)
    if paused:
        raise paused[0]


# ------------------------------------------------------------------ stage 1: anchors (LLM)
def run_anchors(run_id: str, client: LLMClient, s: Settings, progress: Progress, stop: threading.Event | None) -> None:
    eps = load_episodes(run_id)
    todo = []
    for ep in eps:
        out = run_dir(run_id) / "anchors" / f"ep_{ep.episode:03d}.json"
        if out.exists():
            continue
        if not ep.beats:
            write_json(out, {"episode": ep.episode, "model": "", "model_output": None, "accepted": [],
                             "dropped": [], "skipped": "no beats parsed"})
            continue
        todo.append(ep)
    already = len(eps) - len(todo)

    def work(ep: EpisodeScript) -> None:
        try:
            rec = tag_episode(ep, client, s.max_hooks_per_episode, s.max_cliffs_per_episode)
            write_json(run_dir(run_id) / "anchors" / f"ep_{ep.episode:03d}.json", rec)
            with _lock:   # mark done and clear any earlier failure for this episode (resume case)
                st = status(run_id)
                st["anchors_done"] = sorted(set(st.get("anchors_done", [])) | {ep.episode})
                st["anchors_failed"] = {k: v for k, v in st.get("anchors_failed", {}).items() if k != str(ep.episode)}
                write_json(run_dir(run_id) / "status.json", st)
        except QuotaPause:
            raise
        except Exception as e:   # one bad episode never stops the run
            log.error("anchors failed for episode %s: %s\n%s", ep.episode, e, traceback.format_exc())
            set_status(run_id, anchors_failed={str(ep.episode): str(e)[:400]})

    progress("anchors", already, len(eps), f"tagging {len(todo)} episodes, {s.concurrency} at a time")
    _parallel(run_id, todo, work, s.concurrency, stop,
              lambda k: progress("anchors", already + k, len(eps), f"{already + k}/{len(eps)} episodes tagged"),
              label=lambda ep: ep.episode)
    progress("anchors", len(eps), len(eps), "anchor tagging finished")


def load_anchors(run_id: str) -> tuple[list[Anchor], dict[int, str]]:
    anchors, models = [], {}
    for f in sorted((run_dir(run_id) / "anchors").glob("ep_*.json")):
        rec = read_json(f, {}) or {}
        anchors += [Anchor.model_validate(a) for a in rec.get("accepted", [])]
        models[rec.get("episode")] = rec.get("model", "")
    return anchors, models


# ------------------------------------------------------------------ stage 2: windows (code)
def build_windows(run_id: str, s: Settings) -> list[Candidate]:
    anchors, _ = load_anchors(run_id)
    cands, stats = build_candidates(load_episodes(run_id), anchors, per_episode=s.candidates_per_episode)
    write_json(run_dir(run_id) / "candidates.json", {"stats": stats, "candidates": [c.model_dump() for c in cands]})
    return cands


def load_candidates(run_id: str) -> tuple[list[Candidate], dict]:
    rec = read_json(run_dir(run_id) / "candidates.json", {}) or {}
    return [Candidate.model_validate(c) for c in rec.get("candidates", [])], rec.get("stats", {})


# ------------------------------------------------------------------ stage 3: critic (LLM)
def run_critic(run_id: str, cands: list[Candidate], client: LLMClient, s: Settings, progress: Progress,
               stop: threading.Event | None) -> None:
    d = run_dir(run_id) / "critic"
    idx = Index(load_episodes(run_id))
    todo = [c for c in cands if not (d / f"{c.cand_id}.json").exists()]
    with _lock:   # a resumed critic pass starts with a clean failure list
        st = status(run_id)
        st["critic_failed"] = {}
        write_json(run_dir(run_id) / "status.json", st)
    batches = [todo[i:i + s.critic_batch_size] for i in range(0, len(todo), s.critic_batch_size)]

    def work(batch: list[Candidate]) -> None:
        try:
            verdicts = critique_batch(batch, idx, client, f"critic_{batch[0].cand_id}")
            model = client.last_model   # thread-local: the model that answered THIS batch
        except QuotaPause:
            raise
        except Exception as e:
            log.error("critic batch %s failed: %s", batch[0].cand_id, e)
            set_status(run_id, critic_failed={batch[0].cand_id: str(e)[:300]})
            return
        for v in verdicts:
            write_json(d / f"{v.cand_id}.json", {"model": model, "verdict": v.model_dump()})
        missing = {c.cand_id for c in batch} - {v.cand_id for v in verdicts}
        if missing:
            set_status(run_id, critic_failed={m: "critic returned no verdict for this candidate" for m in missing})

    progress("critic", 0, len(batches), f"cold-viewer review: {len(batches)} batches, {s.concurrency} at a time")
    _parallel(run_id, batches, work, s.concurrency, stop,
              lambda k: progress("critic", k, len(batches), f"{k}/{len(batches)} critic batches done"),
              label=lambda b: b[0].cand_id)
    progress("critic", len(batches), len(batches), "critic finished")


# ------------------------------------------------------------------ stage 4: verify + tier (code)
def assemble(run_id: str, cands: list[Candidate] | None = None, eps: list[EpisodeScript] | None = None) -> list[Promo]:
    """Apply saved critic verdicts, verify and tier. No LLM calls, so it is also safe to call while a
    run is still in progress (the live UI does this to show provisional results)."""
    eps = eps if eps is not None else load_episodes(run_id)
    idx = Index(eps)
    if cands is None:
        cands, _ = load_candidates(run_id)
    _, anchor_models = load_anchors(run_id)
    promos = []
    for c in cands:
        rec = read_json(run_dir(run_id) / "critic" / f"{c.cand_id}.json")
        v = CriticVerdictLLM.model_validate(rec["verdict"]) if rec else None
        segs, hook, cliff, changes, issues = apply_verdict(c, v, idx)
        p = Promo(promo_id=c.cand_id, mode=c.mode, segments=segs, original_segments=c.segments, hook_beat=hook,
                  cliff_beat=cliff, hook_intensity=c.hook_intensity, cliffhanger_tension=c.cliffhanger_tension,
                  hook_offset=round(idx.beat[hook].start_sec - segs[0].start_sec, 2),
                  duration=round(sum(x.duration for x in segs), 2), hook_text=idx.beat[hook].text,
                  cliff_text=idx.beat[cliff].text, transcript=clip_lines(segs, idx), verdict=v,
                  boundary_changes=changes, issues=issues,
                  anchor_model=anchor_models.get(idx.beat[c.cliff_beat].episode, ""),
                  critic_model=(rec or {}).get("model", ""))
        p.issues += verify(p, idx)
        promos.append(p)
    return schedule(promos)


def finalize(run_id: str) -> list[Promo]:
    promos = assemble(run_id)
    d = run_dir(run_id) / "results"
    write_json(d / "promos.json", [p.model_dump() for p in promos])
    (d / "posting_schedule.csv").write_text(to_csv(promos), encoding="utf-8")
    (d / "quality_report.md").write_text(quality_report(run_id, promos), encoding="utf-8")
    return promos


def load_promos(run_id: str) -> list[Promo]:
    return [Promo.model_validate(p) for p in read_json(run_dir(run_id) / "results" / "promos.json", [])]


def to_csv(promos: list[Promo]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["post_slot", "tier", "promo_id", "mode", "episodes", "segments", "duration_s", "hook_offset_s",
                "first_3s_hook", "cliffhanger_open", "cold_viewer_clarity", "hook_intensity", "cliffhanger_tension",
                "hook", "cliffhanger", "why_it_works", "tier_reasons", "issues", "boundary_changes"])
    for p in promos:
        v = p.verdict
        w.writerow([p.post_slot or "", p.tier, p.promo_id, p.mode, ";".join(map(str, p.episodes)),
                    " | ".join(f"Ep{s.episode} {fmt(s.start_sec)}-{fmt(s.end_sec)}" for s in p.segments),
                    p.duration, p.hook_offset, v.first_3s_hook if v else "", v.cliffhanger_open if v else "",
                    v.cold_viewer_clarity if v else "", p.hook_intensity, p.cliffhanger_tension,
                    (v.hook_line if v and v.hook_line else p.hook_text), (v.cliffhanger_line if v and v.cliffhanger_line else p.cliff_text),
                    v.why_it_works if v else "", "; ".join(p.tier_reasons),
                    "; ".join(f"{i.severity}:{i.code}" for i in p.issues), "; ".join(p.boundary_changes)])
    return buf.getvalue()


def quality_report(run_id: str, promos: list[Promo]) -> str:
    st = status(run_id)
    _, stats = load_candidates(run_id)
    dropped = []
    for f in sorted((run_dir(run_id) / "anchors").glob("ep_*.json")):
        dropped += (read_json(f, {}) or {}).get("dropped", [])
    anchors, _ = load_anchors(run_id)
    tiers = Counter(p.tier for p in promos)
    codes = Counter((i.code, i.severity, i.source) for p in promos for i in p.issues)
    t3 = Counter(r.split(" (")[0].split(":")[0] for p in promos if p.tier == "TIER_3" for r in p.tier_reasons)
    lines = [f"# Quality report - run `{run_id}`", "",
             "All numbers below are counts from this run. Critic scores are model opinions, not ground truth;",
             "no human has reviewed these promos unless stated.", "",
             "## Coverage",
             f"- Episodes parsed: {len(st.get('episodes', []))}; beats: {st.get('beats')}; parse issues: {st.get('parse_issue_count')}",
             f"- Episodes anchor-tagged: {len(st.get('anchors_done', []))}; failed: {sorted(st.get('anchors_failed', {}))}",
             f"- Anchors accepted: {sum(a.kind == 'HOOK_ANCHOR' for a in anchors)} hooks, "
             f"{sum(a.kind == 'CLIFFHANGER_ANCHOR' for a in anchors)} cliffhangers; "
             f"repointed to a neighbouring beat: {sum(1 for a in anchors if a.repointed_from)}",
             f"- Anchors dropped (not grounded in the screenplay): {len(dropped)}", "",
             "## Windows",
             f"- Mode A (continuous) windows found: {stats.get('mode_a_windows', 0)}; Mode B (stitched): {stats.get('mode_b_windows', 0)}",
             f"- Sent to critic: {stats.get('selected', 0)} ({stats.get('selected_mode_a', 0)} A / {stats.get('selected_mode_b', 0)} B); "
             f"skipped during selection: {stats.get('skipped', {})}",
             f"- Episodes with no candidate: {stats.get('episodes_without_candidates', [])}", "",
             "## Campaign tiers", *[f"- {TIER_LABEL[t]}: {tiers.get(t, 0)}" for t in TIER_LABEL], "",
             "## Where the tool picks badly",
             "Tier 3 reasons (critic opinion):", *([f"- {k}: {n}" for k, n in t3.most_common()] or ["- none"]), "",
             "Issues (code = verified against the screenplay, critic = model opinion):",
             *([f"- `{c}` ({sev}, {src}): {n}" for (c, sev, src), n in codes.most_common()] or ["- none"]), "",
             f"- Boundary trims applied: {sum(1 for p in promos if p.boundary_changes)}",
             f"- Critic failures: {st.get('critic_failed', {})}", "",
             "## Posting order (first 10)", "", "| slot | tier | promo | segments | dur | hook | why |", "|---|---|---|---|---|---|---|"]
    for p in [p for p in promos if p.post_slot][:10]:
        segs = " + ".join(f"Ep{s.episode} {fmt(s.start_sec)}-{fmt(s.end_sec)}" for s in p.segments)
        lines.append(f"| {p.post_slot} | {p.tier} | {p.promo_id} | {segs} | {p.duration:.1f}s | "
                     f"{(p.verdict.hook_line if p.verdict else p.hook_text)[:60]} | {(p.verdict.why_it_works if p.verdict else '')[:80]} |")
    return "\n".join(lines) + "\n"


def raw_outputs_zip(run_id: str) -> bytes:
    """Unedited model outputs plus the per-stage files. The screenplay pack itself is not included."""
    d = run_dir(run_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("raw_outputs", "anchors", "critic", "results"):
            for f in sorted((d / sub).glob("*")):
                if f.is_file():
                    z.write(f, f"{sub}/{f.name}")
        for name in ("candidates.json", "status.json", "run.log"):
            if (d / name).exists():
                z.write(d / name, name)
    return buf.getvalue()


# ------------------------------------------------------------------ everything
def run(run_id: str, settings: Settings | None = None, progress: Progress | None = None,
        stop: threading.Event | None = None, client: LLMClient | None = None) -> list[Promo]:
    s = settings or get_settings()
    progress = progress or (lambda *a: None)
    d = run_dir(run_id)
    handler = logging.FileHandler(d / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("anchorcut").addHandler(handler)
    logging.getLogger("anchorcut").setLevel(logging.INFO)
    client = client or LLMClient(s, d / "raw_outputs")

    def prog(stage, done, total, msg):
        set_status(run_id, progress={"stage": stage, "done": done, "total": total, "message": msg})
        progress(stage, done, total, msg)

    try:
        set_status(run_id, state="running", stage="anchors", last_error=None, model=s.public())
        run_anchors(run_id, client, s, prog, stop)
        if stop is not None and stop.is_set():
            set_status(run_id, state="stopped")
            return []
        set_status(run_id, stage="windows")
        cands = build_windows(run_id, s)
        prog("windows", 1, 1, f"{len(cands)} candidate windows built")
        set_status(run_id, stage="critic")
        run_critic(run_id, cands, client, s, prog, stop)
        if stop is not None and stop.is_set():
            set_status(run_id, state="stopped")
            return []
        set_status(run_id, stage="verify")
        promos = finalize(run_id)
        st = status(run_id)
        complete = not st.get("anchors_failed") and not st.get("critic_failed") and \
            len(st.get("anchors_done", [])) >= len([e for e in load_episodes(run_id) if e.beats])
        set_status(run_id, state="complete" if complete else "partial", stage="done", llm_calls=client.calls)
        prog("done", 1, 1, f"{sum(1 for p in promos if p.post_slot)} promos scheduled")
        return promos
    except QuotaPause as e:
        set_status(run_id, state="paused", last_error=f"{e} Resume later; finished work is kept.")
        try:
            build_windows(run_id, s)
            finalize(run_id)
        except Exception:
            pass
        return []
    except Exception as e:
        set_status(run_id, state="error", last_error=f"{type(e).__name__}: {e}")
        raise
    finally:
        logging.getLogger("anchorcut").removeHandler(handler)
        handler.close()


def rebuild(run_id: str, settings: Settings | None = None) -> list[Promo]:
    """Recompute windows, verification and tiers from saved model outputs. No LLM calls."""
    build_windows(run_id, settings or get_settings())
    return finalize(run_id)
