"""Deterministic window expansion and cross-beat stitching (pure Python, no LLM).

Anchors are single beats. This module turns pairs of them into 20-60 s candidate promos using only
timestamp arithmetic on the parsed screenplay.

Boundary rules (the same for every mode):
  * A clip never starts or ends in the middle of a spoken line. If a DIALOGUE beat is in progress at a
    cut point, the start moves back to where the line begins / the end moves forward to where it ends.
    The end also waits for an ACTION beat in progress to finish. CAMERA and SFX beats may be cut.
  * The hook beat must begin within 3.0 s of the clip start (hook_offset <= 3).
  * Duration = sum over segments of (end - start). It must be 20.0 - 60.0 s inclusive.

Mode A - continuous arc: HOOK_ANCHOR h and a later CLIFFHANGER_ANCHOR c in the same episode,
         clip = [h.start, c.end] (after boundary snapping), kept if 20 <= length <= 60.
Mode B - flash-forward cold open + scene stitch, used where Mode A finds nothing for a hook or a
         cliffhanger: a 3-8 s segment cut around a hook from LATER in the story, followed by a continuous
         stretch that ends on a cliffhanger (same episode, before the hook, or the previous episode).

Selection: candidates get a structural prior (anchor strengths + closeness to a 40 s target, a small
penalty for stitched cuts). Then windows that overlap more than 50 % inside the same scene are
de-duplicated, each anchor beat is reused at most once, and each episode keeps at most N candidates,
so the critic sees diverse moments from every episode instead of 10 variations of the best one.
"""
from __future__ import annotations

from collections import defaultdict

from .models import Anchor, Beat, Candidate, EpisodeScript, Segment

MIN_S, MAX_S, HOOK_S = 20.0, 60.0, 3.0
FF_MIN, FF_MAX = 3.0, 8.0
TARGET_S = 40.0
EPS = 0.05


class Index:
    def __init__(self, episodes: list[EpisodeScript]):
        self.eps = {e.episode: e for e in episodes}
        self.beat = {b.beat_id: b for e in episodes for b in e.beats}

    def beats(self, ep: int) -> list[Beat]:
        return self.eps[ep].beats if ep in self.eps else []

    def beats_in(self, ep: int, start: float, end: float) -> list[Beat]:
        """Beats that begin inside [start, end) - what plays in that segment."""
        return [b for b in self.beats(ep) if start - EPS <= b.start_sec < end - EPS]


# ------------------------------------------------------------------ boundary snapping
def safe_start(beats: list[Beat], start: float, max_back: float) -> float | None:
    """Move a start point back to the beginning of a line in progress. None if that needs > max_back s."""
    for _ in range(4):
        live = [b for b in beats if b.type == "DIALOGUE" and b.start_sec < start - EPS and b.end_sec > start + EPS]
        if not live:
            return round(start, 2)
        start = min(b.start_sec for b in live)
    return None


def safe_end(beats: list[Beat], end: float) -> float:
    """Move an end point forward until no dialogue line or action is cut."""
    for _ in range(8):
        live = [b for b in beats if b.type in ("DIALOGUE", "ACTION") and b.start_sec < end - EPS and b.end_sec > end + EPS]
        if not live:
            break
        end = max(b.end_sec for b in live)
    return round(end, 2)


def is_clean_entry(beats: list[Beat], t: float) -> bool:
    return not any(b.type == "DIALOGUE" and b.start_sec < t - EPS and b.end_sec > t + EPS for b in beats)


def _seg(idx: Index, ep: int, start: float, end: float, role: str) -> Segment:
    inside = idx.beats_in(ep, start, end) or [min(idx.beats(ep), key=lambda b: abs(b.start_sec - start))]
    return Segment(episode=ep, start_sec=round(start, 2), end_sec=round(end, 2),
                   first_beat=inside[0].beat_id, last_beat=inside[-1].beat_id, role=role)


def _prior(h: Anchor, c: Anchor, dur: float, stitched: bool) -> float:
    return round(h.strength + c.strength - abs(dur - TARGET_S) / 10.0 - (0.75 if stitched else 0.0), 3)


# ------------------------------------------------------------------ Mode A
def continuous_arc(idx: Index, h: Anchor, c: Anchor) -> Candidate | None:
    hb, cb = idx.beat[h.beat_id], idx.beat[c.beat_id]
    if hb.episode != cb.episode or cb.order <= hb.order:
        return None
    beats = idx.beats(hb.episode)
    start = safe_start(beats, hb.start_sec, HOOK_S)
    if start is None:
        return None
    end = safe_end(beats, cb.end_sec)
    dur = end - start
    if not (MIN_S <= dur <= MAX_S):
        return None
    return Candidate(
        cand_id=f"A-E{hb.episode:02d}-{hb.order:03d}-{cb.order:03d}", mode="A",
        segments=[_seg(idx, hb.episode, start, end, "arc")], hook_beat=hb.beat_id, cliff_beat=cb.beat_id,
        hook_intensity=h.strength, cliffhanger_tension=c.strength, hook_offset=round(hb.start_sec - start, 2),
        prior=_prior(h, c, dur, False))


# ------------------------------------------------------------------ Mode B
def flash_forward(idx: Index, h: Anchor) -> tuple[float, float] | None:
    """A 3-8 s cold-open segment that starts on the hook beat."""
    hb = idx.beat[h.beat_id]
    beats = idx.beats(hb.episode)
    start = safe_start(beats, hb.start_sec, 1.0)
    if start is None:
        return None
    end = safe_end(beats, hb.end_sec)
    while end - start < FF_MIN:
        nxt = [b.end_sec for b in beats if b.type != "CAMERA" and b.start_sec >= hb.start_sec and b.end_sec > end + EPS]
        if not nxt:
            break
        end = safe_end(beats, min(nxt))
    if not (FF_MIN <= end - start <= FF_MAX):
        return None
    return start, end


def stitched(idx: Index, h: Anchor, c: Anchor) -> Candidate | None:
    hb, cb = idx.beat[h.beat_id], idx.beat[c.beat_id]
    if cb.episode not in (hb.episode, hb.episode - 1):
        return None
    ff = flash_forward(idx, h)
    if ff is None:
        return None
    ff_start, ff_end = ff
    ff_len = ff_end - ff_start
    beats = idx.beats(cb.episode)
    body_end = safe_end(beats, cb.end_sec)
    if cb.episode == hb.episode and body_end > ff_start + EPS:
        return None   # the cold open must come from later in the story than the body it precedes
    lo, hi = MIN_S - ff_len, MAX_S - ff_len
    target = TARGET_S - ff_len

    def entries(pool: list[Beat]) -> list[float]:
        return sorted({b.start_sec for b in pool if b.type != "CAMERA" and b.start_sec < body_end - EPS
                       and lo <= body_end - b.start_sec <= hi and is_clean_entry(beats, b.start_sec)})

    # prefer to stay inside the cliffhanger's scene; fall back to the whole episode
    options = entries([b for b in beats if b.scene == cb.scene]) or entries(beats)
    if not options:
        return None
    body_start = min(options, key=lambda t: abs((body_end - t) - target))
    dur = ff_len + body_end - body_start
    return Candidate(
        cand_id=f"B-{hb.beat_id}-{cb.beat_id}".replace(".", ""), mode="B",
        segments=[_seg(idx, hb.episode, ff_start, ff_end, "flash_forward"),
                  _seg(idx, cb.episode, body_start, body_end, "scene_body")],
        hook_beat=hb.beat_id, cliff_beat=cb.beat_id, hook_intensity=h.strength, cliffhanger_tension=c.strength,
        hook_offset=round(hb.start_sec - ff_start, 2), prior=_prior(h, c, dur, True),
        notes=[f"flash-forward from episode {hb.episode} {ff_len:.1f}s, then episode {cb.episode} body"])


# ------------------------------------------------------------------ dedup + selection
def overlap_share(a: Candidate, b: Candidate, idx: Index) -> float:
    """Shared seconds inside the same scene / length of the shorter candidate."""
    shared = 0.0
    for s in a.segments:
        for t in b.segments:
            if s.episode != t.episode:
                continue
            sa = idx.beat[s.first_beat].scene
            sb = idx.beat[t.first_beat].scene
            if sa != sb:
                continue
            shared += max(0.0, min(s.end_sec, t.end_sec) - max(s.start_sec, t.start_sec))
    return shared / max(min(a.duration, b.duration), 1.0)


def build_candidates(episodes: list[EpisodeScript], anchors: list[Anchor], per_episode: int = 3,
                     overlap_limit: float = 0.5) -> tuple[list[Candidate], dict]:
    """Returns (selected candidates, stats for the quality report)."""
    idx = Index(episodes)
    anchors = [a for a in anchors if a.beat_id in idx.beat]
    hooks = [a for a in anchors if a.kind == "HOOK_ANCHOR"]
    cliffs = [a for a in anchors if a.kind == "CLIFFHANGER_ANCHOR"]
    by_ep_cliffs: dict[int, list[Anchor]] = defaultdict(list)
    for c in cliffs:
        by_ep_cliffs[c.episode].append(c)

    pool: list[Candidate] = []
    hook_has_arc: set[str] = set()
    cliff_has_arc: set[str] = set()
    for h in hooks:
        for c in by_ep_cliffs[h.episode]:
            cand = continuous_arc(idx, h, c)
            if cand:
                pool.append(cand)
                hook_has_arc.add(h.beat_id)
                cliff_has_arc.add(c.beat_id)
    n_arc = len(pool)
    for h in hooks:
        if h.strength < 5:
            continue
        for c in by_ep_cliffs[h.episode] + by_ep_cliffs[h.episode - 1]:
            if c.strength < 5 or (h.beat_id in hook_has_arc and c.beat_id in cliff_has_arc):
                continue   # Mode B only where a single continuous stretch could not use the anchor
            cand = stitched(idx, h, c)
            if cand:
                pool.append(cand)

    pool.sort(key=lambda c: (-c.prior, c.cand_id))
    kept: list[Candidate] = []
    used_hooks: set[str] = set()
    used_cliffs: set[str] = set()
    per_ep: dict[int, int] = defaultdict(int)
    skipped = defaultdict(int)
    for c in pool:
        home = idx.beat[c.cliff_beat].episode
        if per_ep[home] >= per_episode:
            skipped["episode_cap"] += 1
            continue
        if c.hook_beat in used_hooks or c.cliff_beat in used_cliffs:
            skipped["anchor_already_used"] += 1
            continue
        if any(overlap_share(c, k, idx) > overlap_limit for k in kept):
            skipped["overlaps_kept_window"] += 1
            continue
        kept.append(c)
        used_hooks.add(c.hook_beat)
        used_cliffs.add(c.cliff_beat)
        per_ep[home] += 1

    eps_with_anchors = {a.episode for a in anchors}
    stats = {"hook_anchors": len(hooks), "cliffhanger_anchors": len(cliffs), "mode_a_windows": n_arc,
             "mode_b_windows": len(pool) - n_arc, "selected": len(kept),
             "selected_mode_a": sum(c.mode == "A" for c in kept), "selected_mode_b": sum(c.mode == "B" for c in kept),
             "skipped": dict(skipped),
             "episodes_without_candidates": sorted(e for e in idx.eps if per_ep[e] == 0),
             "episodes_without_anchors": sorted(e for e in idx.eps if e not in eps_with_anchors)}
    kept.sort(key=lambda c: (idx.beat[c.cliff_beat].episode, idx.beat[c.segments[0].first_beat].order))
    return kept, stats
