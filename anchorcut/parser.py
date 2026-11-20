"""Deterministic screenplay beat parser (pure Python, no LLM).

Input shape (invented example in the pack's format; parsed tolerantly):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CHAPEL – DAY
    Scene start: 00:00.00
    Scene end: 01:40.00
    Actions and dialogue:
    **CAMERA:**                                <- beat tag
    Start: (00:00.00) | End: (00:04.00)        <- beat timing
    Wide shot of the chapel doors...           <- beat text (may wrap)
    **ANNA (Furious):**
    Start: (00:06.20) | End: (00:09.00)
    "You married my own sister?"
    --- SCRIPT END ---

Nothing is dropped silently: every line the parser cannot place, every beat without
a usable timestamp and every malformed timing line becomes a ParseIssue.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .models import Beat, EpisodeScript, ParseIssue, SceneInfo

TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
RE_HEADER = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_SCENE_FIELD = re.compile(r"^Scene\s+(heading|start|end)\s*:\s*(.*)$", re.I)
RE_BOLD_TAG = re.compile(r"^\*\*\s*(.+?)\s*\*\*\s*:?\s*$")
RE_PLAIN_TAG = re.compile(r"^(CAMERA|ACTION|SOUND FX|SFX|MUSIC)\s*:\s*$", re.I)
RE_TIMING = re.compile(rf"^Start\s*:\s*\(?\s*({TIME})\s*\)?\s*\|?\s*End\s*:\s*\(?\s*({TIME})\s*\)?\s*$", re.I)
RE_FILE_EP = re.compile(r"episode[\s_-]*0*(\d+)", re.I)
STRUCTURAL = re.compile(r"^(---\s*SCRIPT\s+(START|END)\s*---|Actions and dialogue\s*:?)$", re.I)

TYPE_OF_TAG = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX", "MUSIC": "SFX"}


def to_seconds(ts: str) -> float:
    """'02:18.00' -> 138.0 ; '1:02:03.5' -> 3723.5 (MM:SS.cc or HH:MM:SS.cc)."""
    parts = ts.strip().replace(",", ".").split(":")
    if len(parts) not in (2, 3) or not all(p for p in parts):
        raise ValueError(f"bad timestamp {ts!r}")
    secs = float(parts[-1])
    mins = int(parts[-2])
    hours = int(parts[0]) if len(parts) == 3 else 0
    if secs >= 60 or mins >= 60 and len(parts) == 3:
        raise ValueError(f"bad timestamp {ts!r}")
    return round(hours * 3600 + mins * 60 + secs, 3)


def fmt(sec: float) -> str:
    """Seconds -> 'MM:SS.cc', the same notation the screenplays use."""
    sec = max(0.0, round(sec, 2))
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _classify(tag: str) -> tuple[str, str | None, str | None]:
    tag = tag.strip().rstrip(":").strip()
    upper = tag.upper()
    for key, kind in TYPE_OF_TAG.items():
        if upper == key or upper.startswith(key + " ") or upper.startswith(key + "("):
            return kind, None, None
    speaker = tag.split("(", 1)[0].strip().rstrip(":").strip()
    notes = [n.strip() for n in re.findall(r"\(([^)]*)\)", tag) if n.strip()]
    return "DIALOGUE", speaker or tag, ", ".join(notes) or None


def parse_episode(text: str, file: str = "", fallback_number: int = 0) -> EpisodeScript:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = text.split("\n")
    fname = Path(file).name
    issues: list[ParseIssue] = []

    def issue(no: int, line: str, reason: str) -> None:
        issues.append(ParseIssue(file=fname, line_no=no, line=line.strip()[:200], reason=reason))

    # ---- header (fields may wrap onto following lines)
    header: dict[str, str] = {}
    current = None
    body_from = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if RE_SCENE.match(s) or "SCRIPT START" in s.upper():
            body_from = i
            break
        m = RE_HEADER.match(s)
        if m:
            current = m.group(1).upper()
            header[current] = m.group(2).strip()
        elif s and current:
            header[current] += " " + s
        elif s:
            issue(i + 1, raw, "text before the script body that is not a known header field")
        else:
            current = None

    number = None
    if "EPISODE NUMBER" in header:
        m = re.search(r"\d+", header["EPISODE NUMBER"])
        number = int(m.group()) if m else None
    if number is None:
        m = RE_FILE_EP.search(fname)
        number = int(m.group(1)) if m else fallback_number
        issue(0, "", f"episode number missing from header; using {number} (from "
                     f"{'filename' if m else 'file order'})")

    ep = EpisodeScript(episode=number, file=fname, logline=header.get("EPISODE LOGLINE", ""),
                       continuity=header.get("CONTINUITY NOTES", ""))
    scene: SceneInfo | None = None
    pending: dict | None = None   # beat being assembled

    def close_beat() -> None:
        nonlocal pending
        if pending is None:
            return
        p, pending = pending, None
        body = " ".join(p["text"]).strip()
        if p["type"] == "DIALOGUE":
            body = body.strip('"“” ').strip()
        if p["start"] is None:
            issue(p["line_no"], p["tag_line"], "beat has no usable timestamp; beat skipped")
            return
        if p["end"] < p["start"]:
            issue(p["timing_line_no"], p["timing_line"], "end time is before start time; beat skipped")
            return
        if not body:
            issue(p["line_no"], p["tag_line"], "beat has a timestamp but no text (kept)")
        order = len(ep.beats)
        ep.beats.append(Beat(
            beat_id=f"E{number:02d}.B{order:03d}", episode=number, scene=scene.number if scene else 0,
            order=order, start_sec=p["start"], end_sec=p["end"], type=p["type"], speaker=p["speaker"],
            direction=p["direction"], text=body, raw_start=p["raw_start"], raw_end=p["raw_end"]))

    for i in range(body_from, len(lines)):
        raw = lines[i]
        s = raw.strip()
        no = i + 1
        if not s:
            continue
        if STRUCTURAL.match(s):
            if "END" in s.upper() and "SCRIPT" in s.upper():
                close_beat()
            continue
        m = RE_SCENE.match(s)
        if m:
            close_beat()
            scene = SceneInfo(number=int(m.group(1)))
            ep.scenes.append(scene)
            continue
        m = RE_SCENE_FIELD.match(s)
        if m and pending is None:
            if scene is None:
                issue(no, raw, "scene field before any SCENE marker")
                continue
            field, val = m.group(1).lower(), m.group(2).strip()
            if field == "heading":
                scene.heading = val
            else:
                try:
                    setattr(scene, f"{field}_sec", to_seconds(val))
                except ValueError:
                    issue(no, raw, f"malformed scene {field} timestamp")
            continue
        m = RE_BOLD_TAG.match(s) or RE_PLAIN_TAG.match(s)
        if m:
            close_beat()
            kind, speaker, direction = _classify(m.group(1))
            pending = dict(type=kind, speaker=speaker, direction=direction, text=[], start=None, end=None,
                           raw_start="", raw_end="", line_no=no, tag_line=raw, timing_line_no=no, timing_line="")
            if scene is None:
                issue(no, raw, "beat appears before any SCENE marker (assigned to scene 0)")
            continue
        if s.lower().startswith("start") and pending is not None and pending["start"] is None and not pending["text"]:
            m = RE_TIMING.match(s)
            if not m:
                issue(no, raw, "malformed timing line")
                continue
            try:
                st, en = to_seconds(m.group(1)), to_seconds(m.group(2))
            except ValueError as e:
                issue(no, raw, f"malformed timestamp: {e}")
                continue
            pending.update(start=st, end=en, raw_start=m.group(1), raw_end=m.group(2),
                           timing_line_no=no, timing_line=raw)
            continue
        if pending is not None:
            pending["text"].append(s)
        else:
            issue(no, raw, "line outside any beat; ignored")
    close_beat()

    ends = [b.end_sec for b in ep.beats] + [sc.end_sec for sc in ep.scenes if sc.end_sec is not None]
    ep.duration_sec = round(max(ends), 2) if ends else 0.0
    if not ep.scenes:
        issue(0, "", "no SCENE markers found")
    if not ep.beats:
        issue(0, "", "no timestamped beats found")
    ep.issues = issues
    return ep


def load_pack(data: bytes | str | Path) -> tuple[list[EpisodeScript], list[ParseIssue]]:
    """Parse every .txt screenplay in a ZIP. Returns (episodes sorted by number, pack-level issues)."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError("the upload is not a valid ZIP file") from e
    pack_issues: list[ParseIssue] = []
    episodes: list[EpisodeScript] = []
    with zf:
        names = sorted(n for n in zf.namelist() if not n.endswith("/"))
        for k, name in enumerate(names, 1):
            base = Path(name).name
            if "__MACOSX" in name or base.startswith("._"):
                continue
            if not base.lower().endswith(".txt"):
                pack_issues.append(ParseIssue(file=base, line_no=0, line="", reason="not a .txt file; ignored"))
                continue
            raw = zf.read(name)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="replace")
                pack_issues.append(ParseIssue(file=base, line_no=0, line="", reason="not UTF-8; decoded as cp1252"))
            episodes.append(parse_episode(text, name, fallback_number=k))
    seen: dict[int, str] = {}
    for e in episodes:
        if e.episode in seen:
            pack_issues.append(ParseIssue(file=e.file, line_no=0, line="",
                                          reason=f"duplicate episode number {e.episode} (also in {seen[e.episode]})"))
        seen[e.episode] = e.file
    episodes.sort(key=lambda e: e.episode)
    return episodes, pack_issues
