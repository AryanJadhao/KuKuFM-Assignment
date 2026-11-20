"""Offline tests. A synthetic screenplay (no confidential text) and a scripted fake LLM that exists only
in this file - it is never used to produce results."""
import io
import json
import zipfile

import pytest

# 60 s episode: camera opener, a slap (hook) at 4 s, lines, a threat at ~33 s (cliffhanger), more scenes.
EP = """EPISODE NUMBER: Episode {n}
EPISODE LOGLINE: A bride is abandoned at the altar.
CONTINUITY NOTES: Synthetic fixture.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CHAPEL – DAY
Scene start: 00:00.00
Scene end: 00:40.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the chapel doors.

**ACTION:**
Start: (00:04.00) | End: (00:06.00)
Anna slaps Tom hard across the face.

**ANNA (Furious) (Shaking):**
Start: (00:06.20) | End: (00:09.00)
"You married my own sister behind my back?"

**TOM (Cold):**
Start: (00:09.50) | End: (00:20.00)
"I never loved you, and everyone in this chapel knew it."

**CAMERA**
Start: 00:20.00 | End: 00:30.00
A stranger in black steps out of a carriage.

**STRANGER (O.S.):**
Start: (00:30.00) | End: (00:36.00)
"Then marry me instead, unless you want to die tonight."

**SOUND FX:**
Start: (00:36.00) | End: (00:40.00)
Thunder.

SCENE 02
Scene heading: EXT. ROAD – NIGHT
Scene start: 00:40.00
Scene end: 01:00.00

**ANNA (Whispering):**
Start: (00:40.00) | End: (00:52.00)
"Who are you, and why do you know my name?"

**STRANGER:**
Start: (00:52.50) | End: (00:58.00)
"I am the man your family tried to bury."

**CAMERA:**
Start: (00:58.00) | End: (01:00.00)
Cut to black.
"""


def pack(n=2, extra=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k in range(1, n + 1):
            z.writestr(f"show/episode-{k:03d}-x.txt", EP.format(n=k))
        z.writestr("__MACOSX/._junk.txt", "junk")
        for name, text in (extra or {}).items():
            z.writestr(name, text)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ANCHORCUT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LLM_RPM", "100000")


# ------------------------------------------------------------------ parser
def test_parser_beats_and_timestamps():
    from anchorcut.parser import fmt, load_pack, to_seconds
    eps, issues = load_pack(pack(1))
    assert issues == [] and len(eps) == 1
    ep = eps[0]
    assert [b.type for b in ep.beats] == ["CAMERA", "ACTION", "DIALOGUE", "DIALOGUE", "CAMERA", "DIALOGUE", "SFX",
                                          "DIALOGUE", "DIALOGUE", "CAMERA"]
    anna = ep.beats[2]
    assert anna.beat_id == "E01.B002" and anna.speaker == "ANNA" and anna.direction == "Furious, Shaking"
    assert anna.text.startswith("You married") and anna.start_sec == 6.2 and anna.raw_start == "00:06.20"
    assert ep.beats[4].start_sec == 20.0          # unbolded tag, unparenthesised timing
    assert ep.beats[7].scene == 2 and ep.duration_sec == 60.0 and ep.issues == []
    assert to_seconds("01:02.50") == 62.5 and to_seconds("1:00:00.0") == 3600 and fmt(62.5) == "01:02.50"


def test_parser_logs_malformed_lines_instead_of_dropping_silently():
    from anchorcut.parser import parse_episode
    bad = EP.format(n=3).replace("Start: (00:04.00) | End: (00:06.00)", "Start: (00:04.0O) | End: (00:06.00)")
    bad = bad.replace("**TOM (Cold):**\nStart: (00:09.50) | End: (00:20.00)", "**TOM (Cold):**")
    ep = parse_episode(bad, "episode-003.txt")
    reasons = " | ".join(i.reason for i in ep.issues)
    assert "malformed timing line" in reasons and "no usable timestamp" in reasons
    assert len(ep.beats) == 8


def test_invalid_zip_rejected():
    from anchorcut.parser import load_pack
    with pytest.raises(ValueError):
        load_pack(b"not a zip")


# ------------------------------------------------------------------ anchors + grounding
def test_anchor_validation_grounds_and_repoints():
    from anchorcut.anchor_tagger import validate_sheet
    from anchorcut.models import AnchorSheetLLM
    from anchorcut.parser import load_pack
    ep = load_pack(pack(1))[0][0]
    sheet = AnchorSheetLLM.model_validate({
        "hook_anchors": [
            {"beat_id": "E01.B001", "hook_intensity": 12, "excerpt": "Anna slaps Tom hard"},              # clamp to 10
            {"beat_id": "E01.B000", "hook_intensity": 6, "excerpt": "You married my own sister"},        # repoint -> B002
            {"beat_id": "E01.B003", "hook_intensity": 9, "excerpt": "You stole my crown and my horse"},  # invented
            {"beat_id": "E01.B099", "hook_intensity": 9, "excerpt": "x"}],                              # bad id
        "cliffhanger_anchors": [{"beat_id": "E01.B005", "cliffhanger_tension": 8, "excerpt": "unless you want to die tonight"}]})
    acc, dropped = validate_sheet(ep, sheet)
    by = {(a.kind, a.beat_id): a for a in acc}
    assert by[("HOOK_ANCHOR", "E01.B001")].strength == 10
    assert by[("HOOK_ANCHOR", "E01.B002")].repointed_from == "E01.B000"
    assert {d["reason"] for d in dropped} == {"excerpt not found in the screenplay text", "beat_id not in this episode"}


# ------------------------------------------------------------------ window builder
def _anchor(kind, bid, s):
    from anchorcut.models import Anchor
    return Anchor(kind=kind, beat_id=bid, episode=int(bid[1:3]), strength=s, excerpt="x")


def test_mode_a_arithmetic_and_no_mid_line_cut():
    from anchorcut.parser import load_pack
    from anchorcut.window_builder import Index, continuous_arc
    idx = Index(load_pack(pack(1))[0])
    c = continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B005", 8))
    assert c and c.mode == "A"
    s = c.segments[0]
    assert (s.start_sec, s.end_sec) == (4.0, 36.0) and c.duration == 32.0 and c.hook_offset == 0
    # cliffhanger on the 12 s line in scene 2: end snaps to the line end, 4.0 -> 52.0 = 48 s
    c2 = continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B007", 8))
    assert c2.segments[0].end_sec == 52.0 and c2.duration == 48.0
    # too short (< 20 s) and wrong order are refused
    assert continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B002", 8)) is None
    assert continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B005", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B001", 8)) is None


def test_mode_b_flash_forward_stitch():
    from anchorcut.parser import load_pack
    from anchorcut.window_builder import FF_MAX, FF_MIN, Index, stitched
    idx = Index(load_pack(pack(2))[0])
    # hook from episode 2 (later), body ends on episode 1's threat
    c = stitched(idx, _anchor("HOOK_ANCHOR", "E02.B008", 8), _anchor("CLIFFHANGER_ANCHOR", "E01.B005", 9))
    assert c and c.mode == "B" and [s.role for s in c.segments] == ["flash_forward", "scene_body"]
    ff, body = c.segments
    assert FF_MIN <= ff.duration <= FF_MAX and (ff.episode, body.episode) == (2, 1)
    assert body.end_sec == 36.0 and 20 <= c.duration <= 60
    # a cold open from EARLIER in the story is not a flash-forward
    assert stitched(idx, _anchor("HOOK_ANCHOR", "E01.B001", 8), _anchor("CLIFFHANGER_ANCHOR", "E02.B005", 9)) is None


def test_dedup_and_episode_cap():
    from anchorcut.parser import load_pack
    from anchorcut.window_builder import build_candidates
    eps = load_pack(pack(1))[0]
    anchors = [_anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("HOOK_ANCHOR", "E01.B002", 8),
               _anchor("CLIFFHANGER_ANCHOR", "E01.B005", 8), _anchor("CLIFFHANGER_ANCHOR", "E01.B006", 7)]
    kept, stats = build_candidates(eps, anchors, per_episode=3)
    assert stats["mode_a_windows"] >= 3
    assert len(kept) == 1      # the other windows overlap the same scene by > 50 %
    assert stats["skipped"]


# ------------------------------------------------------------------ verify + tiers
def _promo(idx, cand, verdict=None):
    from anchorcut.cold_viewer_critic import apply_verdict, clip_lines
    from anchorcut.models import Promo
    segs, hook, cliff, changes, issues = apply_verdict(cand, verdict, idx)
    return Promo(promo_id=cand.cand_id, mode=cand.mode, segments=segs, original_segments=cand.segments, hook_beat=hook,
                 cliff_beat=cliff, hook_intensity=cand.hook_intensity, cliffhanger_tension=cand.cliffhanger_tension,
                 hook_offset=idx.beat[hook].start_sec - segs[0].start_sec, duration=sum(s.duration for s in segs),
                 hook_text="", cliff_text="", transcript=clip_lines(segs, idx), verdict=verdict,
                 boundary_changes=changes, issues=issues)


def _verdict(cid, h, c, k, **kw):
    from anchorcut.models import CriticVerdictLLM
    return CriticVerdictLLM.model_validate({"cand_id": cid, "first_3s_hook": h, "cliffhanger_open": c,
                                           "cold_viewer_clarity": k, "hook_line": "Anna slaps Tom", **kw})


def test_verify_blocks_bad_windows():
    from anchorcut.models import Segment
    from anchorcut.parser import load_pack
    from anchorcut.verify import verify
    from anchorcut.window_builder import Index, continuous_arc
    idx = Index(load_pack(pack(1))[0])
    c = continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B005", 8))
    assert [i for i in verify(_promo(idx, c), idx) if i.severity == "block"] == []
    # hand-made broken window: starts mid-line at 12 s, hook beat 8 s earlier, too short
    c.segments = [Segment(episode=1, start_sec=12.0, end_sec=25.0, first_beat="E01.B004", last_beat="E01.B004", role="arc")]
    codes = {i.code for i in verify(_promo(idx, c), idx)}
    assert {"duration_out_of_range", "cuts_mid_line", "hook_not_in_opening"} <= codes


def test_trims_applied_only_when_safe_and_tiers():
    from anchorcut.cold_viewer_critic import schedule
    from anchorcut.parser import load_pack
    from anchorcut.verify import verify
    from anchorcut.window_builder import Index, continuous_arc
    idx = Index(load_pack(pack(1))[0])
    c = continuous_arc(idx, _anchor("HOOK_ANCHOR", "E01.B001", 9), _anchor("CLIFFHANGER_ANCHOR", "E01.B007", 8))
    # trimming the end to the threat line keeps it 32 s -> applied; cliffhanger moves
    p = _promo(idx, c, _verdict(c.cand_id, 9, 8, 9, trim_end_at="E01.B005", trim_reason="next line answers it"))
    assert p.cliff_beat == "E01.B005" and p.segments[0].end_sec == 36.0 and p.boundary_changes
    # trimming the start to the stranger's threat would leave < 20 s -> discarded
    p2 = _promo(idx, c, _verdict(c.cand_id, 9, 8, 9, trim_start_to="E01.B007"))
    assert not p2.boundary_changes and any(i.code == "trim_discarded" for i in p2.issues)
    p.issues += verify(p, idx)
    p3 = _promo(idx, c, _verdict("x3", 9, 9, 4))                 # low clarity -> Tier 3
    p3.promo_id = "x3"
    p4 = _promo(idx, c, _verdict("x4", 7, 8, 7, ambiguity=[{"kind": "pronoun", "beat_id": "E01.B003"}]))
    p4.promo_id = "x4"
    out = schedule([p3, p4, p])
    tiers = {q.promo_id: q.tier for q in out}
    assert tiers[p.promo_id] == "TIER_1" and tiers["x3"] == "TIER_3" and tiers["x4"] == "TIER_2"
    assert [q.post_slot for q in out] == [1, 2, None]


# ------------------------------------------------------------------ end to end with a scripted fake model
class FakeLLM:
    """Test double. Returns canned JSON keyed on the schema it is asked for."""
    calls = 0
    last_model = "fake:test"

    def ask_json(self, prompt, schema, tag):
        from anchorcut.models import AnchorSheetLLM, CriticBatchLLM
        self.calls += 1
        if schema is AnchorSheetLLM:
            e = f"E{int(tag[len('anchors_ep'):len('anchors_ep') + 3]):02d}"
            return AnchorSheetLLM.model_validate({
                "hook_anchors": [{"beat_id": f"{e}.B001", "hook_intensity": 9, "excerpt": "Anna slaps Tom hard across the face"},
                                 {"beat_id": f"{e}.B008", "hook_intensity": 8, "excerpt": "the man your family tried to bury"}],
                "cliffhanger_anchors": [{"beat_id": f"{e}.B005", "cliffhanger_tension": 9, "excerpt": "unless you want to die tonight"},
                                        {"beat_id": f"{e}.B007", "cliffhanger_tension": 6, "excerpt": "why do you know my name"}]})
        if schema is CriticBatchLLM:
            ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("### CLIP")]
            return CriticBatchLLM.model_validate({"verdicts": [
                {"cand_id": i, "first_3s_hook": 9 if i.startswith("A") else 6, "cliffhanger_open": 8, "cold_viewer_clarity": 8,
                 "hook_line": "Anna slaps Tom hard across the face.", "cliffhanger_line": "unless you want to die tonight",
                 "viewer_takeaway": "t", "why_it_works": "w"} for i in ids]})
        raise AssertionError(schema)


def test_pipeline_end_to_end_and_resume():
    from anchorcut import pipeline
    run_id = pipeline.ingest(pack(2), "test")
    fake = FakeLLM()
    promos = pipeline.run(run_id, client=fake)
    st = pipeline.status(run_id)
    assert st["state"] == "complete" and st["anchors_done"] == [1, 2]
    assert promos and all(20 <= p.duration <= 60 for p in promos if p.tier != "REJECTED")
    assert any(p.tier == "TIER_1" for p in promos) and promos[0].post_slot == 1
    d = pipeline.run_dir(run_id)
    for f in ("results/promos.json", "results/posting_schedule.csv", "results/quality_report.md", "candidates.json"):
        assert (d / f).exists(), f
    z = zipfile.ZipFile(io.BytesIO(pipeline.raw_outputs_zip(run_id)))
    assert any(n.startswith("anchors/") for n in z.namelist()) and not any(n.endswith(".txt") for n in z.namelist())
    # resume: everything is on disk, so no model calls
    again = FakeLLM()
    pipeline.run(run_id, client=again)
    assert again.calls == 0
    assert json.loads((d / "results" / "promos.json").read_text(encoding="utf-8"))


def test_llm_client_saves_raw_and_reasks(tmp_path, monkeypatch):
    from anchorcut.config import get_settings
    from anchorcut.llm_client import LLMClient
    from anchorcut.models import AnchorSheetLLM
    s = get_settings()
    s.api_key, s.provider, s.model, s.fallback_models = "SECRET-KEY-123", "gemini", "m1", ""
    replies = iter(["not json at all", '```json\n{"hook_anchors": [], "cliffhanger_anchors": []}\n```'])
    c = LLMClient(s, tmp_path / "raw")
    monkeypatch.setattr(c, "_post", lambda model, prompt: next(replies))
    out = c.ask_json("prompt", AnchorSheetLLM, "anchors_ep001")
    assert out.hook_anchors == []
    files = sorted(p.name for p in (tmp_path / "raw").iterdir())
    assert files == ["anchors_ep001__1.json", "anchors_ep001__2.json"]
    first = json.loads((tmp_path / "raw" / files[0]).read_text(encoding="utf-8"))
    assert first["raw_response"] == "not json at all" and first["parse_error"]
    assert not any("SECRET-KEY-123" in f.read_text(encoding="utf-8") for f in (tmp_path / "raw").iterdir())


def test_quota_moves_to_fallback_model_then_pauses(tmp_path, monkeypatch):
    import httpx
    from anchorcut.config import get_settings
    from anchorcut.llm_client import LLMClient, QuotaPause
    from anchorcut.models import AnchorSheetLLM
    s = get_settings()
    s.api_key, s.provider, s.model, s.fallback_models = "k", "gemini", "big", "small"
    served = []

    def fake_post(url, **kw):
        served.append(url.split("/models/")[1].split(":")[0])
        req = httpx.Request("POST", url)
        if "big" in url:
            return httpx.Response(429, text='{"error": {"details": [{"retryDelay": "40000s"}]}}', request=req)
        return httpx.Response(200, request=req, json={"candidates": [{"content": {"parts": [
            {"text": '{"hook_anchors": [], "cliffhanger_anchors": []}'}]}}]})
    monkeypatch.setattr(httpx, "post", fake_post)
    c = LLMClient(s, None)
    c.ask_json("p", AnchorSheetLLM, "t1")
    c.ask_json("p", AnchorSheetLLM, "t2")
    assert served == ["big", "small", "small"] and c.last_model == "gemini:small"
    s.fallback_models = ""
    with pytest.raises(QuotaPause):
        LLMClient(s, None).ask_json("p", AnchorSheetLLM, "t3")


def test_overload_fails_one_call_without_pausing_run(monkeypatch):
    import time as _time
    import httpx
    from anchorcut.config import get_settings
    from anchorcut.llm_client import LLMClient, LLMError, QuotaPause
    from anchorcut.models import AnchorSheetLLM
    s = get_settings()
    s.api_key, s.provider, s.model, s.fallback_models, s.max_retries = "k", "gemini", "m1", "", 2
    monkeypatch.setattr(_time, "sleep", lambda *_: None)
    state = {"down": True}

    def fake_post(url, **kw):
        req = httpx.Request("POST", url)
        if state["down"]:
            return httpx.Response(503, text="overloaded", request=req)
        return httpx.Response(200, request=req, json={"candidates": [{"content": {"parts": [
            {"text": '{"hook_anchors": [], "cliffhanger_anchors": []}'}]}}]})
    monkeypatch.setattr(httpx, "post", fake_post)
    c = LLMClient(s, None)
    with pytest.raises(LLMError) as ei:
        c.ask_json("p", AnchorSheetLLM, "t1")
    assert not isinstance(ei.value, QuotaPause)          # unit fails, run keeps going
    state["down"] = False
    assert c.ask_json("p", AnchorSheetLLM, "t2").hook_anchors == []   # same model still usable afterwards


# ------------------------------------------------------------------ swappable model / keys
def _clear_llm_env(monkeypatch):
    import os
    for k in [k for k in os.environ if k.startswith(("LLM_", "GEMINI_", "GOOGLE_", "OPENROUTER_", "OPENAI_", "GROQ_"))]:
        monkeypatch.delenv(k, raising=False)
    import anchorcut.config as cfg
    monkeypatch.setattr(cfg, "_load_env_file", lambda: None)   # isolate from the developer's real .env


def test_each_provider_uses_its_own_key_and_default_model(monkeypatch):
    _clear_llm_env(monkeypatch)
    from anchorcut.config import get_settings
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    s = get_settings()
    assert (s.provider, s.api_key, s.model) == ("gemini", "gem-key", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    s = get_settings()
    assert (s.api_key, s.model, s.endpoint) == ("or-key", "google/gemini-2.5-flash", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    s = get_settings()
    assert s.api_key == "" and any("no API key for groq" in p for p in s.problems())   # never borrows another key
    monkeypatch.setenv("LLM_API_KEY", "generic")
    assert get_settings().api_key == "generic"


def test_ui_overrides_swap_provider_model_and_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    from anchorcut.config import get_settings
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    base = get_settings()
    s = base.with_overrides(provider="openrouter", model="anthropic/claude-x")
    assert (s.provider, s.model, s.api_key, s.fallback_models) == ("openrouter", "anthropic/claude-x", "or-key", "")
    s2 = base.with_overrides(provider="openai", model="m", base_url="https://gw.example/v1", api_key="typed")
    assert (s2.endpoint, s2.api_key, s2.key_source.startswith("entered")) == ("https://gw.example/v1", "typed", True)
    assert "typed" not in str(s2.public()) and base.api_key == "gem-key"   # base untouched, key never displayed


def test_dotenv_changes_apply_without_restart(tmp_path, monkeypatch):
    import os
    import anchorcut.config as cfg
    for k in [k for k in os.environ if k.startswith(("LLM_", "GEMINI_", "GOOGLE_", "OPENROUTER_", "OPENAI_", "GROQ_"))]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cfg, "ROOT", tmp_path)               # point the loader at a throwaway .env, never the real one
    monkeypatch.setattr(cfg, "_from_dotenv", set())
    tag = lambda s: s.api_key if s.api_key in ("first", "second", "from-shell") else "<other>"  # never echo real keys
    (tmp_path / ".env").write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=first\n", encoding="utf-8")
    assert tag(cfg.get_settings()) == "first"
    (tmp_path / ".env").write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=second\nLLM_MODEL=other-model\n",
                                   encoding="utf-8")
    s = cfg.get_settings()
    assert (tag(s), s.model) == ("second", "other-model")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")       # a real env var (e.g. Streamlit secret) still wins
    cfg._from_dotenv.discard("GEMINI_API_KEY")
    assert tag(cfg.get_settings()) == "from-shell"
