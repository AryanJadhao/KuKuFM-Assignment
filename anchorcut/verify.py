"""Deterministic verification - the 'Verify' in Anchor -> Expand -> Verify.

Every check here is computed from the parsed screenplay, never from model text, and is labelled
source="code". Blocking issues remove a promo from the schedule; warnings stay visible on the card.
"""
from __future__ import annotations

from .models import Issue, Promo
from .text_match import GROUNDED, grounded_ratio
from .window_builder import EPS, HOOK_S, MAX_S, MIN_S, Index

CLIFF_TAIL_S = 2.5


def _i(code: str, sev: str, detail: str) -> Issue:
    return Issue(code=code, severity=sev, detail=detail, source="code")


def verify(p: Promo, idx: Index) -> list[Issue]:
    out: list[Issue] = []
    if not p.segments:
        return [_i("no_segments", "block", "candidate has no segments")]
    if not (MIN_S - 1e-6 <= p.duration <= MAX_S + 1e-6):
        out.append(_i("duration_out_of_range", "block", f"total {p.duration:.2f}s is outside 20-60s"))

    for k, s in enumerate(p.segments, 1):
        ep = idx.eps.get(s.episode)
        if ep is None or s.start_sec < 0 or s.end_sec <= s.start_sec or s.end_sec > ep.duration_sec + EPS:
            out.append(_i("bad_timestamps", "block", f"segment {k} ({s.start_sec}-{s.end_sec}s) is not inside episode {s.episode}"))
            continue
        beats = ep.beats
        for t, edge in ((s.start_sec, "start"), (s.end_sec, "end")):
            cut = [b for b in beats if b.type == "DIALOGUE" and b.start_sec < t - EPS and b.end_sec > t + EPS]
            if cut:
                out.append(_i("cuts_mid_line", "block", f"segment {k} {edge} cuts {cut[0].beat_id} mid-line"))
    for a in range(len(p.segments)):
        for b in range(a + 1, len(p.segments)):
            x, y = p.segments[a], p.segments[b]
            if x.episode == y.episode and x.start_sec < y.end_sec - EPS and y.start_sec < x.end_sec - EPS:
                out.append(_i("segments_overlap", "block", f"segments {a + 1} and {b + 1} repeat the same footage"))

    hb = idx.beat.get(p.hook_beat)
    first = p.segments[0]
    if hb is None or hb.episode != first.episode or not (first.start_sec - EPS <= hb.start_sec < first.end_sec):
        out.append(_i("hook_not_in_opening", "block", f"hook beat {p.hook_beat} is not in the first segment"))
    elif hb.start_sec - first.start_sec > HOOK_S + EPS:
        out.append(_i("hook_too_late", "block", f"hook starts {hb.start_sec - first.start_sec:.1f}s in (limit 3s)"))

    cb = idx.beat.get(p.cliff_beat)
    last = p.segments[-1]
    if cb is None or cb.episode != last.episode or not (last.start_sec - EPS <= cb.start_sec < last.end_sec):
        out.append(_i("cliffhanger_not_in_ending", "block", f"cliffhanger beat {p.cliff_beat} is not in the last segment"))
    elif cb.end_sec < last.end_sec - CLIFF_TAIL_S:
        out.append(_i("plays_past_cliffhanger", "warn",
                      f"clip runs {last.end_sec - cb.end_sec:.1f}s after the cliffhanger beat"))

    if p.verdict:
        text = "\n".join(p.transcript)
        for label, q in (("hook_line", p.verdict.hook_line), ("cliffhanger_line", p.verdict.cliffhanger_line)):
            if q.strip() and grounded_ratio(q, text) < GROUNDED:
                out.append(_i("unsupported_quote", "warn", f"critic {label} is not in the clip: \"{q[:70]}\""))
        for a in p.verdict.ambiguity:
            if a.beat_id and a.beat_id.strip().upper() not in {line.split()[1] for line in p.transcript if line.startswith("t=")}:
                out.append(_i("ambiguity_outside_clip", "warn", f"critic flagged {a.beat_id}, which is not in the clip"))
    return out
