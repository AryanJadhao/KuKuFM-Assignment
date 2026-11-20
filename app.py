"""AnchorCut - Editorial Storyboard & Timeline Inspector.

    streamlit run app.py

The page imports the pipeline directly (single process). Long LLM work runs in a background thread
and reports through status.json, so changing filters or views never re-runs a model call.
"""
from __future__ import annotations

import html
import os
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AnchorCut · Promo Storyboard", page_icon="✂️", layout="wide")

# Streamlit Cloud: secrets -> env (never displayed). Locally .env is read by config.py.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str) and not os.getenv(k):
            os.environ[k] = v
except Exception:
    pass

from anchorcut import jobs, live, pipeline  # noqa: E402
from anchorcut.config import get_settings  # noqa: E402
from anchorcut.models import TIER_LABEL, Promo  # noqa: E402
from anchorcut.parser import fmt  # noqa: E402
from anchorcut.window_builder import Index  # noqa: E402

INK, PAPER, RULE = "#1E1B18", "#FFFFFF", "#E4DED3"
HOOK_C, BODY_C, CLIFF_C, CUT_C, MUTED = "#E8A33D", "#C9D3DC", "#B4441E", "#1E1B18", "#7A746B"
TIER_C = {"TIER_1": "#2F6B4F", "TIER_2": "#3B5B92", "TIER_3": "#B4441E", "REJECTED": "#7A746B", "UNREVIEWED": "#A39E95"}

st.markdown(f"""<style>
.block-container {{padding-top: 1.4rem; max-width: 1280px;}}
h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}}
.ac-brand {{font-family: Georgia, serif; font-size: 1.55rem; font-weight: 700; color:{INK}; line-height:1.1}}
.ac-sub {{color:{MUTED}; font-size: .82rem; margin-bottom: .6rem}}
.ac-card {{background:{PAPER}; border:1px solid {RULE}; border-radius:10px; padding:14px 16px 10px; margin-bottom:12px;}}
.ac-head {{display:flex; gap:10px; align-items:baseline; flex-wrap:wrap}}
.ac-slot {{font-family: Georgia, serif; font-size:1.5rem; font-weight:700; min-width:2.2rem; color:{INK}}}
.ac-id {{font-family: ui-monospace, Consolas, monospace; font-size:.8rem; color:{MUTED}}}
.ac-chip {{display:inline-block; font-size:.72rem; padding:1px 8px; border-radius:999px; border:1px solid {RULE}; color:{INK}; background:#F7F4EE; margin-right:4px}}
.ac-chip.warn {{border-color:#E8A33D; background:#FDF4E3}} .ac-chip.block {{border-color:#B4441E; background:#FBE9E3}}
.ac-chip.critic {{border-style:dashed}}
.ac-tl {{position:relative; height:26px; background:#F4F1EB; border-radius:5px; margin:10px 0 2px; overflow:hidden}}
.ac-tl div {{position:absolute; top:0; height:100%}}
.ac-ticks {{position:relative; height:14px; font-size:.65rem; color:{MUTED}}}
.ac-ticks span {{position:absolute; transform:translateX(-50%)}}
.ac-q {{font-family: Georgia, serif; font-style: italic; color:{INK}; margin:2px 0 6px}}
.ac-lbl {{font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}}}
.ac-seg {{font-family: ui-monospace, Consolas, monospace; font-size:.8rem}}
.ac-score {{display:inline-block; text-align:center; margin-right:14px}}
.ac-score b {{font-family: Georgia, serif; font-size:1.25rem}}
.ac-score small {{display:block; font-size:.66rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.05em}}
</style>""", unsafe_allow_html=True)

def effective_settings():
    """.env / secrets, plus any model or key chosen in the 'Model & API key' panel (this session only)."""
    cfg = st.session_state.get("model_cfg") or {}
    return get_settings().with_overrides(**cfg)


S = effective_settings()
READY = not S.problems()


# ------------------------------------------------------------------ cached loaders (keyed on file mtime)
def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=8)
def _episodes(run_id: str, mtime: float):  # mtime is part of the cache key (no leading _)
    return pipeline.load_episodes(run_id)


@st.cache_data(show_spinner=False, max_entries=8)
def _promos(run_id: str, mtime: float):  # mtime is part of the cache key (no leading _)
    return pipeline.load_promos(run_id)


def episodes(run_id):
    return _episodes(run_id, _mtime(pipeline.run_dir(run_id) / "parsed" / "episodes.json"))


def promos(run_id) -> list[Promo]:
    return _promos(run_id, _mtime(pipeline.run_dir(run_id) / "results" / "promos.json"))


# ------------------------------------------------------------------ small renderers
def timeline_html(p: Promo, idx: Index, scale: float = 60.0) -> str:
    """Clip drawn on a fixed 60 s ruler: amber = first 3 s, slate = body, rust = cliffhanger beat, black = cut."""
    parts, t0 = [], 0.0
    pct = lambda x: f"{100 * x / scale:.2f}%"  # noqa: E731
    for k, s in enumerate(p.segments):
        parts.append(f'<div style="left:{pct(t0)};width:{pct(s.duration)};background:{BODY_C}"></div>')
        if k:
            parts.append(f'<div style="left:{pct(t0)};width:2px;background:{CUT_C}" title="hard cut"></div>')
        t0 += s.duration
    parts.append(f'<div style="left:0;width:{pct(min(3.0, p.duration))};background:{HOOK_C}" title="first 3 seconds"></div>')
    cb = idx.beat.get(p.cliff_beat)
    last = p.segments[-1]
    if cb:
        rel_start = p.duration - last.duration + max(0.0, cb.start_sec - last.start_sec)
        rel_end = min(p.duration, p.duration - last.duration + cb.end_sec - last.start_sec)
        parts.append(f'<div style="left:{pct(rel_start)};width:{pct(max(rel_end - rel_start, 0.4))};background:{CLIFF_C}" title="cliffhanger beat"></div>')
    band = f'<div style="left:{pct(20)};width:{pct(40)};border-top:2px dashed #CFC8BB;height:0;top:0"></div>'
    ticks = "".join(f'<span style="left:{pct(t)}">{t}s</span>' for t in (0, 3, 20, 40, 60))
    return f'<div class="ac-tl">{"".join(parts)}{band}</div><div class="ac-ticks">{ticks}</div>'


def chips(p: Promo) -> str:
    out = [f'<span class="ac-chip">{"continuous arc" if p.mode == "A" else "flash-forward stitch"}</span>',
           f'<span class="ac-chip">Ep {", ".join(map(str, p.episodes))}</span>',
           f'<span class="ac-chip">{p.duration:.1f}s</span>']
    if p.boundary_changes:
        out.append('<span class="ac-chip" title="critic trim applied">trimmed</span>')
    for i in p.issues:
        cls = f"ac-chip {i.severity}" + (" critic" if i.source == "critic" else "")
        out.append(f'<span class="{cls}" title="{html.escape(i.detail)} ({i.source})">{html.escape(i.code)}</span>')
    return "".join(out)


def scores_html(p: Promo) -> str:
    v = p.verdict
    if not v:
        return f'<span class="ac-lbl">not reviewed by critic</span>'
    cells = [("first 3s hook", v.first_3s_hook), ("cliffhanger", v.cliffhanger_open), ("cold clarity", v.cold_viewer_clarity),
             ("hook anchor", p.hook_intensity), ("tension anchor", p.cliffhanger_tension)]
    return "".join(f'<span class="ac-score"><b>{x:.0f}</b><small>{lbl}</small></span>' for lbl, x in cells)


def card(p: Promo, idx: Index, key: str) -> None:
    v = p.verdict
    hook = v.hook_line if v and v.hook_line else p.hook_text
    cliff = v.cliffhanger_line if v and v.cliffhanger_line else p.cliff_text
    segs = "<br>".join(f'<span class="ac-seg">{k}. Episode {s.episode} · {fmt(s.start_sec)} → {fmt(s.end_sec)} · {s.duration:.1f}s'
                       f' · {s.role.replace("_", " ")}</span>' for k, s in enumerate(p.segments, 1))
    st.markdown(f"""<div class="ac-card">
<div class="ac-head"><span class="ac-slot">{('#' + str(p.post_slot)) if p.post_slot else '–'}</span>
<span class="ac-id">{p.promo_id}</span> {chips(p)}</div>
{timeline_html(p, idx)}
<div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:6px">
 <div style="flex:3;min-width:280px">
  <div class="ac-lbl">Hook · opens at t={p.hook_offset:.1f}s</div><div class="ac-q">“{html.escape(hook)}”</div>
  <div class="ac-lbl">Cuts on</div><div class="ac-q">“{html.escape(cliff)}”</div>
  <div class="ac-lbl">Why it works</div><div>{html.escape(v.why_it_works if v else '—')}</div>
 </div>
 <div style="flex:2;min-width:240px">
  <div class="ac-lbl">Segments, in play order</div>{segs}
  <div style="margin-top:8px">{scores_html(p)}</div>
  <div class="ac-lbl" style="margin-top:4px">Tier reasoning</div><div style="font-size:.85rem">{html.escape('; '.join(p.tier_reasons))}</div>
 </div>
</div></div>""", unsafe_allow_html=True)
    st.button("Open in Timeline Inspector →", key=f"ins_{key}", type="tertiary", on_click=_open_inspector, args=(p.promo_id,))


def _open_inspector(promo_id: str) -> None:
    # callbacks run before widgets are rebuilt, so the view switch is allowed here
    st.session_state["inspect"] = promo_id
    st.session_state["view"] = "Timeline Inspector"


def busy_elsewhere(rid: str, fresh_s: float = 600) -> bool:
    """status.json says 'running' and was updated recently, but not by a thread in this app."""
    from datetime import datetime, timezone
    s = pipeline.status(rid)
    if s.get("state") != "running" or not s.get("updated_utc"):
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(s["updated_utc"])).total_seconds()
    except ValueError:
        return False
    return age < fresh_s


def model_panel() -> None:
    """Swap provider / model / key from the UI. Applies to runs started from this browser session; nothing is
    written to disk and the key is never displayed again. Leave a field blank to keep the .env / secrets value."""
    from anchorcut.config import PROVIDERS
    base = get_settings()
    cfg = st.session_state.get("model_cfg") or {}
    with st.expander("Model & API key", expanded=bool(S.problems())):
        names = list(PROVIDERS)
        cur = cfg.get("provider") or base.provider
        with st.form("model_form", border=False):
            prov = st.selectbox("Provider", names, index=names.index(cur) if cur in names else 0,
                                help="gemini = Google AI Studio. openrouter / groq / openai = any OpenAI-compatible API.")
            model = st.text_input("Model", value=cfg.get("model", ""),
                                  placeholder=f"default: {PROVIDERS[prov][0] or 'required'}" if prov != base.provider
                                  else f"default: {base.model}")
            fb = st.text_input("Fallback models (comma separated)", value=cfg.get("fallback_models", base.fallback_models
                                                                                    if prov == base.provider else ""))
            url = st.text_input("Base URL (OpenAI-compatible only)", value=cfg.get("base_url", ""),
                                placeholder=PROVIDERS[prov][1] or "not used for gemini")
            key = st.text_input("API key", type="password", value="",
                                placeholder="leave blank to use .env / secrets",
                                help="Kept in this browser session's memory only. Never saved, logged or shown.")
            a, b = st.columns(2)
            apply = a.form_submit_button("Apply", type="primary", use_container_width=True)
            reset = b.form_submit_button("Reset", use_container_width=True)
        if apply:
            new = {"provider": prov, "model": model.strip(), "fallback_models": fb.strip(), "base_url": url.strip()}
            new["api_key"] = key.strip() or (cfg.get("api_key", "") if prov == cfg.get("provider") else "")
            st.session_state["model_cfg"] = new
            st.rerun()
        if reset:
            st.session_state.pop("model_cfg", None)
            st.rerun()
        if st.button("Test connection", use_container_width=True, disabled=bool(S.problems())):
            st.session_state["conn_test"] = test_connection(S)
        if st.session_state.get("conn_test"):
            ok, msg = st.session_state["conn_test"]
            (st.success if ok else st.error)(msg)
        st.caption("Changes apply to the next Run / resume. A run already in progress keeps its model.")


def test_connection(s) -> tuple[bool, str]:
    """One tiny request to the configured model. The key is never echoed in the result."""
    import pydantic
    from anchorcut.llm_client import LLMClient

    class Ping(pydantic.BaseModel):
        ok: bool

    try:
        trial = s.with_overrides()
        trial.max_retries = 1
        c = LLMClient(trial, None)
        c.ask_json('Reply with exactly this JSON: {"ok": true}', Ping, "connection_test")
        return True, f"Connected: {c.last_model} answered."
    except Exception as e:  # noqa: BLE001 - show the reason, never the key
        msg = str(e).replace(s.api_key, "***") if s.api_key else str(e)
        return False, f"Failed: {msg[:300]}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="ac-brand">✂ AnchorCut</div><div class="ac-sub">Anchor → Expand → Verify · promo finder for microdrama screenplays</div>',
                unsafe_allow_html=True)
    runs = pipeline.list_runs()
    with st.expander("Import a screenplay pack (.zip)", expanded=not runs):
        up = st.file_uploader("ZIP of .txt episode screenplays", type=["zip"], label_visibility="collapsed")
        label = st.text_input("Run label", placeholder="e.g. duke-s1")
        auto = st.checkbox("Start finding promos right after parsing", value=READY, disabled=not READY,
                           help="Parsing only reads the screenplays. Finding promos calls the model (~1 min per call).")
        if up is not None and st.button("Parse pack" + (" & start" if auto else ""), type="primary", use_container_width=True):
            data = up.getvalue()
            existing = pipeline.find_run_for_pack(data)
            if existing:
                # same ZIP already imported: open that run instead of creating a duplicate (and paying twice)
                st.session_state["run_id"] = existing
                st.session_state["flash"] = f"This pack was already imported as run {existing}; opened it instead of parsing it again."
                if auto and not jobs.is_running(existing) and not busy_elsewhere(existing) \
                        and pipeline.status(existing).get("state") not in ("complete",):
                    jobs.start(existing, S)
                st.rerun()
            try:
                rid = pipeline.ingest(data, label or up.name.rsplit(".", 1)[0])
                st.session_state["run_id"] = rid
                if auto:
                    jobs.start(rid, S)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    runs = pipeline.list_runs()
    if not runs:
        st.info("Import a ZIP of screenplays to begin.")
        st.stop()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    cur = st.session_state.get("run_id")
    if cur not in runs:   # first visit: open the run that is in progress, else the one with the most results
        cur = next((r for r in runs if jobs.is_running(r) or busy_elsewhere(r)), None) or \
            max(runs, key=lambda r: (pipeline.status(r).get("state") in ("complete", "partial"),
                                     len(pipeline.status(r).get("anchors_done", [])), r))
    run_id = st.selectbox("Run", runs, index=runs.index(cur) if cur in runs else 0,
                          format_func=lambda r: f"{r} · {pipeline.status(r).get('state', '?')}")
    st.session_state["run_id"] = run_id
    running = jobs.is_running(run_id)
    external = not running and busy_elsewhere(run_id)
    if external:
        st.info("This run is being processed by another process (e.g. the CLI). Progress updates live here; "
                "Run is disabled to avoid duplicate model calls.")
    c1, c2 = st.columns(2)
    if c1.button("▶ Run / resume", disabled=running or external or not READY, use_container_width=True, type="primary"):
        jobs.start(run_id, S)
        st.session_state["was_running"] = True
        st.rerun()
    if c2.button("■ Stop", disabled=not running, use_container_width=True):
        jobs.stop(run_id)
    if st.button("↻ Rebuild tiers (no LLM calls)", disabled=running, use_container_width=True):
        pipeline.rebuild(run_id)
        st.rerun()
    pub = S.public()
    st.markdown(f"**Model** `{pub['provider']}` · " + " → ".join(f"`{m}`" for m in pub["model_chain"]))
    st.caption("API key: " + (f"configured ✓ ({pub['key_source']})" if pub["api_key_set"] else "missing"))
    for prob in S.problems():
        st.warning(prob)
    model_panel()


eps = episodes(run_id)
idx = Index(eps)
ps: list[Promo] = []
state: dict = {}
LIVE = running or external
VIEWS = ["Posting Schedule", "Timeline Inspector", "Episode Anchor Map", "Quality", "Export"]
CELL = {"done": ("#2F6B4F", "#FFFFFF"), "working": (HOOK_C, INK), "failed": (CLIFF_C, "#FFFFFF"), "pending": ("#ECE7DE", MUTED)}


@st.cache_data(show_spinner=False, max_entries=4)
def _live_promos(run_id: str, sig: tuple, _eps):  # sig = cache key; _eps is not hashed
    return live.provisional_promos(run_id, _eps)


def _dur(s: float | None) -> str:
    if s is None:
        return "estimating..."
    m, sec = divmod(int(s), 60)
    return f"~{m} min {sec:02d} s" if m else f"~{sec} s"


def live_panel(snap: dict) -> None:
    pr = snap["progress"]
    stage = {"anchors": "Step 1 · tagging anchors", "windows": "Step 2 · building windows",
             "critic": "Step 3 · cold-viewer critic", "verify": "Step 4 · verify + tier"}.get(snap["stage"], snap["stage"])
    tot = max(int(pr.get("total") or 1), 1)
    frac = min(int(pr.get("done") or 0) / tot, 1.0)
    cells = "".join(
        f'<span title="Episode {n}: {s}" style="display:inline-flex;align-items:center;justify-content:center;width:26px;'
        f'height:22px;margin:2px;border-radius:4px;font-size:.66rem;background:{CELL[s][0]};color:{CELL[s][1]};'
        f'{"outline:2px solid " + INK + ";" if s == "working" else ""}">{n}</span>'
        for n, s in snap["episodes"].items())
    feed = "".join(f'<div style="font-size:.8rem;padding:2px 0;border-bottom:1px solid {RULE}"><span class="ac-id">{t}</span> '
                   f'{html.escape(m)}</div>' for t, m in snap["events"]) or '<div class="ac-lbl">waiting for the first result...</div>'
    stale = snap["status_age_s"] is not None and snap["status_age_s"] > 240
    st.markdown(f"""<div class="ac-card" style="border-left:4px solid {HOOK_C}">
<div class="ac-head"><span style="font-family:Georgia,serif;font-size:1.1rem;font-weight:700">● Live · {html.escape(stage)}</span>
<span class="ac-id">{html.escape(pr.get('message', ''))}</span></div>
<div style="height:8px;background:#F0ECE4;border-radius:4px;margin:8px 0 10px"><div style="height:8px;width:{100 * frac:.1f}%;background:{HOOK_C};border-radius:4px"></div></div>
<div style="display:flex;gap:26px;flex-wrap:wrap">
 <div style="flex:3;min-width:300px"><div class="ac-lbl">Episodes · {snap['episodes_done']}/{snap['episodes_total']} tagged</div>{cells}
  <div style="margin-top:8px">
   <span class="ac-score"><b>{snap['hooks']}</b><small>hook anchors</small></span>
   <span class="ac-score"><b>{snap['cliffs']}</b><small>cliff anchors</small></span>
   <span class="ac-score"><b>{snap['dropped']}</b><small>dropped (ungrounded)</small></span>
   <span class="ac-score"><b>{snap['reviewed']}/{snap['candidates'] or '–'}</b><small>critic reviewed</small></span>
   <span class="ac-score"><b>{snap['model_calls']}</b><small>model calls</small></span>
   <span class="ac-score"><b>{snap['retries']}</b><small>re-asks</small></span>
  </div>
  <div class="ac-lbl" style="margin-top:4px">Time left {_dur(snap['eta_s'])} · model {html.escape(', '.join(snap['models']) or '-')}</div>
 </div>
 <div style="flex:2;min-width:260px"><div class="ac-lbl">Latest activity</div>{feed}</div>
</div></div>""", unsafe_allow_html=True)
    if stale:
        st.warning("No progress written for over 4 minutes. The process may be waiting on a rate limit or may have stopped; "
                   "if it stopped, use Run / resume (finished work is kept).")


@st.fragment(run_every=3.0 if LIVE else None)
def body():
    global ps, state
    state = pipeline.status(run_id)
    still_live = jobs.is_running(run_id) or busy_elsewhere(run_id)
    if LIVE and not still_live:
        st.rerun(scope="app")          # run just finished: refresh sidebar buttons and final results
    sig = live.signature(run_id)
    ps = _live_promos(run_id, sig, eps) if still_live else promos(run_id)
    st.markdown(f"<h1 style='margin-bottom:0'>Promo Storyboard</h1><div class='ac-sub'>Run <code>{run_id}</code> · "
                f"{len(eps)} episodes · {state.get('beats', 0)} beats · state <b>{state.get('state', '-')}</b>"
                f"{' · updating every 3 s' if still_live else ''}</div>", unsafe_allow_html=True)
    if still_live:
        live_panel(live.snapshot(run_id, [e.episode for e in eps]))
        st.caption("Results below are provisional and fill in as each step finishes. Windows appear as soon as an "
                   "episode is tagged (in Unreviewed); tiers appear as the critic reviews them.")
    elif state.get("last_error"):
        st.warning(state["last_error"])
    tiers = Counter(p.tier for p in ps)
    m = st.columns(5)
    m[0].metric("Tier 1 · post first", tiers.get("TIER_1", 0))
    m[1].metric("Tier 2 · second wave", tiers.get("TIER_2", 0))
    m[2].metric("Tier 3 · needs a human", tiers.get("TIER_3", 0))
    m[3].metric("Unreviewed / rejected", f"{tiers.get('UNREVIEWED', 0)} / {tiers.get('REJECTED', 0)}")
    m[4].metric("Anchor-tagged episodes", f"{len(state.get('anchors_done', []))}/{len(eps)}")
    st.session_state.setdefault("view", "Posting Schedule")
    view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or "Posting Schedule"
    {"Posting Schedule": view_schedule, "Timeline Inspector": view_inspector, "Episode Anchor Map": view_anchor_map,
     "Quality": view_quality, "Export": view_export}[view]()


# ------------------------------------------------------------------ views
def view_schedule():
    if not ps:
        st.info("No promos yet. Click **Run / resume** in the sidebar. The parsed screenplay is already browsable in "
                "**Episode Anchor Map**.")
        return
    f1, f2 = st.columns([3, 1])
    ep_pick = f1.multiselect("Episodes", sorted({e for p in ps for e in p.episodes}), placeholder="All episodes")
    mode = f2.selectbox("Cut style", ["All", "Continuous arc (A)", "Flash-forward stitch (B)"])
    sel = [p for p in ps if (not ep_pick or set(p.episodes) & set(ep_pick))
           and (mode == "All" or p.mode == mode[-2])]
    order = ["TIER_1", "TIER_2", "TIER_3", "REJECTED", "UNREVIEWED"]
    short = {"TIER_1": "Tier 1 · Cold traffic", "TIER_2": "Tier 2 · Escalation", "TIER_3": "Tier 3 · Flagged for review",
             "REJECTED": "Rejected", "UNREVIEWED": "Unreviewed"}
    tabs = st.tabs([f"{short[t]} ({sum(p.tier == t for p in sel)})" for t in order])
    blurbs = {"TIER_1": "Strongest first-3-second jolt and needs no prior context. Post these to cold traffic first.",
              "TIER_2": "Real drama but a softer hook, softer cliffhanger or small ambiguity. Second-wave posts.",
              "TIER_3": "High tension, but the critic says a stranger may be confused or the ending lets the air out. "
                        "Not scheduled until an editor clears them. This is where the tool knows it might be wrong.",
              "REJECTED": "Failed a code check against the screenplay (duration, hook position, mid-line cut...).",
              "UNREVIEWED": "Built by the window builder but not yet seen by the critic (still running, paused, or a failed batch)."}
    for tab, t in zip(tabs, order):
        with tab:
            st.caption(blurbs[t])
            items = [p for p in sel if p.tier == t]
            for p in items[:80]:
                card(p, idx, f"{t}_{p.promo_id}")
            if len(items) > 80:
                st.caption(f"Showing 80 of {len(items)} - narrow with the episode filter.")


def view_inspector():
    if not ps:
        st.info("Nothing to inspect yet.")
        return
    ids = [p.promo_id for p in ps]
    by = {p.promo_id: p for p in ps}
    cur = st.session_state.get("inspect")
    pid = st.selectbox("Promo", ids, index=ids.index(cur) if cur in ids else 0,
                       format_func=lambda i: f"{('#' + str(by[i].post_slot)) if by[i].post_slot else '  –'} · {i} · {TIER_LABEL[by[i].tier]}")
    st.session_state["inspect"] = pid
    p = by[pid]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(timeline_html(p, idx), unsafe_allow_html=True)
        st.plotly_chart(beat_lanes(p), use_container_width=True, key=f"lanes_{pid}")
    with right:
        st.markdown(f"#### {TIER_LABEL[p.tier]}")
        st.markdown(scores_html(p), unsafe_allow_html=True)
        st.caption("Critic scores are model opinions. Anchor strengths come from the anchor tagger.")
        for r in p.tier_reasons:
            st.markdown(f"- {r}")
        if p.verdict:
            st.markdown(f"**A stranger thinks:** {p.verdict.viewer_takeaway}")
            if p.verdict.ambiguity:
                st.markdown("**Ambiguity flagged by critic**")
                for a in p.verdict.ambiguity:
                    st.markdown(f"- `{a.kind}` at `{a.beat_id}` - {a.detail}")
        st.markdown("**Boundaries**")
        for k, (o, n) in enumerate(zip(p.original_segments, p.segments), 1):
            moved = "" if (o.start_sec, o.end_sec) == (n.start_sec, n.end_sec) else f"  (built as {fmt(o.start_sec)}→{fmt(o.end_sec)})"
            st.markdown(f"<span class='ac-seg'>{k}. Ep {n.episode} {fmt(n.start_sec)} → {fmt(n.end_sec)}{moved}</span>",
                        unsafe_allow_html=True)
        for c in p.boundary_changes:
            st.caption("✂ " + c)
        if p.issues:
            st.markdown("**Checks**")
            for i in p.issues:
                st.markdown(f"- {'⛔' if i.severity == 'block' else '⚠️'} `{i.code}` ({i.source}) {i.detail}")
        st.caption(f"Anchors by `{p.anchor_model or '-'}` · critic `{p.critic_model or '-'}`")
    st.markdown("##### Clip as it plays")
    rows = []
    for line in p.transcript:
        if not line.startswith("t="):
            rows.append({"t": "", "beat": "", "role": "", "text": line.strip("= ")})
            continue
        t, bid, rest = line.split(" ", 2)
        role = "HOOK" if bid == p.hook_beat else ("CLIFF" if bid == p.cliff_beat else "")
        rows.append({"t": t[2:], "beat": bid, "role": role, "text": rest.replace("   <- inside first 3 seconds", "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"text": st.column_config.TextColumn(width="large")})


def beat_lanes(p: Promo) -> go.Figure:
    lanes = {"CAMERA": "Shot", "ACTION": "Action", "DIALOGUE": "Line", "SFX": "Sound"}
    fig = go.Figure()
    t0 = 0.0
    for k, s in enumerate(p.segments):
        for b in idx.beats_in(s.episode, s.start_sec, s.end_sec):
            x0 = t0 + max(0.0, b.start_sec - s.start_sec)
            x1 = t0 + min(s.end_sec, b.end_sec) - s.start_sec
            color = CLIFF_C if b.beat_id == p.cliff_beat else HOOK_C if b.beat_id == p.hook_beat else \
                {"CAMERA": "#D9D3C7", "ACTION": "#9DB0C2", "DIALOGUE": "#5C7A99", "SFX": "#C8B79A"}[b.type]
            who = f"{b.speaker}: " if b.speaker else ""
            fig.add_trace(go.Bar(x=[max(x1 - x0, 0.25)], base=[x0], y=[lanes[b.type]], orientation="h",
                                 marker_color=color, showlegend=False,
                                 hovertemplate=f"{b.beat_id} · t={x0:.1f}s<br>{html.escape(who + b.text[:90])}<extra></extra>"))
        if k:
            fig.add_vline(x=t0, line_width=2, line_color=CUT_C)
        t0 += s.duration
    fig.add_vrect(x0=0, x1=3, fillcolor=HOOK_C, opacity=0.12, line_width=0, annotation_text="first 3 s",
                  annotation_position="top left")
    fig.update_layout(barmode="overlay", height=230, margin=dict(l=0, r=10, t=24, b=10), plot_bgcolor=PAPER,
                      paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="seconds into promo", range=[0, max(t0, 20)], gridcolor=RULE),
                      yaxis=dict(categoryorder="array", categoryarray=["Sound", "Shot", "Action", "Line"]))
    return fig


def view_anchor_map():
    ep_num = st.selectbox("Episode", [e.episode for e in eps], format_func=lambda n: f"Episode {n}")
    ep = idx.eps[ep_num]
    st.caption(ep.logline)
    rec = pipeline.read_json(pipeline.run_dir(run_id) / "anchors" / f"ep_{ep_num:03d}.json", {}) or {}
    acc = rec.get("accepted", [])
    fig = go.Figure()
    for b in ep.beats:
        fig.add_trace(go.Scatter(x=[b.start_sec, b.end_sec], y=[0, 0], mode="lines", line=dict(color="#D9D3C7", width=6),
                                 showlegend=False, hoverinfo="skip"))
    for kind, sym, col, name in (("HOOK_ANCHOR", "triangle-up", HOOK_C, "hook anchor"),
                                 ("CLIFFHANGER_ANCHOR", "diamond", CLIFF_C, "cliffhanger anchor")):
        pts = [a for a in acc if a["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=[idx.beat[a["beat_id"]].start_sec for a in pts], y=[a["strength"] for a in pts], mode="markers",
            marker=dict(symbol=sym, size=13, color=col, line=dict(color=INK, width=1)), name=name,
            text=[f"{a['beat_id']} · {a['strength']}/10<br>{html.escape(a['excerpt'][:80])}<br><i>{html.escape(a['label'])}</i>" for a in pts],
            hovertemplate="%{text}<extra></extra>"))
    here = [p for p in ps if ep_num in p.episodes]
    for k, p in enumerate(here):
        for s in p.segments:
            if s.episode == ep_num:
                fig.add_trace(go.Scatter(x=[s.start_sec, s.end_sec], y=[-1.2 - 0.7 * k] * 2, mode="lines",
                                         line=dict(color=TIER_C[p.tier], width=8), showlegend=False,
                                         hovertemplate=f"{p.promo_id} · {TIER_LABEL[p.tier]}<extra></extra>"))
    fig.update_layout(height=320 + 18 * len(here), margin=dict(l=0, r=10, t=10, b=10), plot_bgcolor=PAPER,
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08),
                      xaxis=dict(title="episode time (s)", gridcolor=RULE),
                      yaxis=dict(title="anchor strength", range=[-1.6 - 0.7 * len(here), 10.8], gridcolor=RULE))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey line = the episode's beats. Markers = anchors the tagger marked (height = strength). "
               "Coloured bars below = promo windows from this episode, coloured by tier.")
    if rec.get("dropped"):
        with st.expander(f"Anchors dropped by grounding check ({len(rec['dropped'])})"):
            st.dataframe(pd.DataFrame(rec["dropped"]), hide_index=True, use_container_width=True)
    with st.expander("Read the screenplay"):
        marks = {a["beat_id"]: ("HOOK " if a["kind"] == "HOOK_ANCHOR" else "CLIFF ") + str(a["strength"]) for a in acc}
        st.dataframe(pd.DataFrame([{"time": f"{b.raw_start}-{b.raw_end}", "beat": b.beat_id, "scene": b.scene,
                                    "type": b.type, "who": b.speaker or "", "text": b.text, "anchor": marks.get(b.beat_id, "")}
                                   for b in ep.beats]), hide_index=True, use_container_width=True, height=420,
                     column_config={"text": st.column_config.TextColumn(width="large")})


def view_quality():
    rep = pipeline.run_dir(run_id) / "results" / "quality_report.md"
    if ps:
        df = pd.DataFrame([{"tier": TIER_LABEL[p.tier], "mode": "A continuous" if p.mode == "A" else "B stitched"} for p in ps])
        fig = go.Figure()
        for mode, col in (("A continuous", "#5C7A99"), ("B stitched", HOOK_C)):
            sub = df[df["mode"] == mode]["tier"].value_counts()
            fig.add_trace(go.Bar(y=list(sub.index), x=list(sub.values), name=mode, orientation="h", marker_color=col))
        fig.update_layout(barmode="stack", height=260, margin=dict(l=0, r=10, t=10, b=10), plot_bgcolor=PAPER,
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.info("The quality report is written when a run finishes (or after **Rebuild tiers**).")
    issues = pipeline.read_json(pipeline.run_dir(run_id) / "parsed" / "parse_issues.json", [])
    with st.expander(f"Parse issues ({len(issues)})"):
        if issues:
            st.dataframe(pd.DataFrame(issues), hide_index=True, use_container_width=True)
        else:
            st.caption("Every timestamp line in the pack was parsed.")


def view_export():
    d = pipeline.run_dir(run_id) / "results"
    c = st.columns(4)
    c[0].download_button("Raw outputs (.zip)", pipeline.raw_outputs_zip(run_id), f"anchorcut_{run_id}_raw_outputs.zip",
                         "application/zip", use_container_width=True, type="primary")
    for col, name, mime in ((c[1], "posting_schedule.csv", "text/csv"), (c[2], "promos.json", "application/json"),
                            (c[3], "quality_report.md", "text/markdown")):
        if (d / name).exists():
            col.download_button(name, (d / name).read_bytes(), f"anchorcut_{run_id}_{name}", mime, use_container_width=True)
    st.caption("raw_outputs/ holds every prompt and the model's reply exactly as received, one file per attempt. "
               "anchors/ and critic/ hold the validated model output per episode / candidate. results/ is what code "
               "derived from them. The input screenplays are never included.")
    with st.expander("Run status (status.json)"):
        st.json(state)


body()
