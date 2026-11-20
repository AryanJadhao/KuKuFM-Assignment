"""LLM step 2 - cold-viewer self-containment critic, then the campaign tier ranker.

The critic reads each assembled candidate exactly as a stranger would experience it (t = seconds from
the clip start, segments in play order) and returns three 1-10 judgements, concrete ambiguity flags,
and optional boundary trims (by beat_id). It never sees the anchor strengths or why the clip was built,
so it cannot be anchored by the earlier step.

Code then:
  1. applies the trims only if the clip stays 20-60 s and the beats are really inside the clip,
  2. verifies the result against the screenplay (verify.py),
  3. places the promo in a campaign tier with explicit, readable rules (no weighted sum):

     TIER_1  cold-traffic acquisition   hook >= 8, clarity >= 8, cliffhanger >= 7, and no relationship /
                                         prior-event confusion flagged
     TIER_3  human review flagged       clarity <= 5; or a relationship / prior-event / speaker ambiguity
                                         together with clarity <= 6; or the ending resolves the tension;
                                         or cliffhanger <= 4; or hook <= 4
     TIER_2  high-stakes escalation      everything else that passed verification

  Calibration note (v2, after the first full run): v1 sent every clip with ANY relationship / prior-event
  note to Tier 3. The critic adds such a note to almost every clip of a mid-series microdrama, even at
  clarity 7-8, so 81% of reviewed clips were flagged and the review queue was meaningless. Ambiguity now
  only flags a clip when the critic also rates clarity <= 6. This was tuned on the same run's verdicts
  (no held-out set), which is stated in the README.

  4. orders a posting schedule: Tier 1 first (sorted by hook, clarity, cliffhanger), then Tier 2
     (sorted by cliffhanger, hook, clarity), never two consecutive posts from the same episode
     when another option exists.
"""
from __future__ import annotations

from .llm_client import LLMClient
from .models import Candidate, CriticBatchLLM, CriticVerdictLLM, Issue, Promo, Segment
from .parser import fmt
from .window_builder import MAX_S, MIN_S, Index, safe_end, safe_start, _seg

PROMPT_VERSION = "critic-v1"
SEVERE_AMBIGUITY = {"relationship", "prior_event", "speaker_identity"}
AMBIGUITY_FLAG_CLARITY = 6                       # severe ambiguity flags a clip only at clarity <= this
TIER1_BLOCKERS = {"relationship", "prior_event"}  # Tier 1 = zero prior context needed
TYPE_WORD = {"CAMERA": "SHOT", "ACTION": "ACTION", "SFX": "SOUND", "DIALOGUE": "LINE"}


# ------------------------------------------------------------------ rendering
def clip_lines(segments: list[Segment], idx: Index) -> list[str]:
    out, t0 = [], 0.0
    for k, s in enumerate(segments, 1):
        label = {"arc": "continuous", "flash_forward": "cold open (flash-forward)", "scene_body": "main scene"}[s.role]
        out.append(f"== segment {k}/{len(segments)} - {label} - episode {s.episode} ==")
        for b in idx.beats_in(s.episode, s.start_sec, s.end_sec):
            rel = t0 + max(0.0, b.start_sec - s.start_sec)
            body = f"{b.speaker}" + (f" ({b.direction})" if b.direction else "") + f': "{b.text}"' \
                if b.type == "DIALOGUE" else b.text
            mark = "   <- inside first 3 seconds" if rel < 3.0 else ""
            out.append(f"t={rel:05.1f} {b.beat_id} {TYPE_WORD[b.type]}: {body}{mark}")
        t0 += s.duration
    out.append(f"t={t0:05.1f} == CLIP ENDS ==")
    return out


def critic_prompt(cands: list[Candidate], idx: Index) -> str:
    blocks = []
    for c in cands:
        blocks.append(f"### CLIP {c.cand_id} ({c.duration:.1f} s, {len(c.segments)} segment(s))\n" +
                      "\n".join(clip_lines(c.segments, idx)))
    ids = ", ".join(c.cand_id for c in cands)
    return f"""You are two people at once: a first-time viewer scrolling a vertical-video feed who has NEVER
seen this show and knows none of its characters, and the ad buyer who decides which clips get paid reach.
Each clip below is written out exactly as it would play (t = seconds since the clip began; a new
segment means a hard cut). Speaker names are production notes: a real viewer only learns a name if
someone says it out loud or it is obvious from the picture.

For every clip, answer honestly and use the full 1-10 range (most clips are not a 9):
- first_3s_hook: judged ONLY on what happens between t=0 and t=3. 1 = nothing would stop a scroll.
- cliffhanger_open: at the final cut, how badly does the viewer need to know what happens next?
  1 = everything is settled.
- cold_viewer_clarity: can a stranger tell who these people are to each other and what is at stake,
  from this clip alone? 1 = lost.
- ambiguity: every concrete confusion, each with kind = pronoun | relationship | unexplained_name |
  prior_event | speaker_identity, the beat_id where it happens, and a short detail. Empty list if none.
- ending_resolves_tension: true if the last beats answer or defuse the tension.
- Boundary fixes (optional; only beat_ids printed inside that clip):
  trim_start_to = the beat that should open the clip if the current first 3 seconds are weak
                  (must be in segment 1); trim_end_at = the beat the clip should cut on if the current
                  ending lets the air out (must be in the last segment). Use null when the boundary is right.
  Clips must stay 20-60 seconds; a trim that breaks that is discarded. Give trim_reason when you trim.
- hook_line and cliffhanger_line: copy the exact words from the clip text (no beat id, no speaker).
- viewer_takeaway: one sentence - what a stranger thinks this story is about.
- why_it_works: one line for the ad buyer.

{chr(10).join(blocks)}

Return ONLY JSON with one verdict per clip ({ids}):
{{"verdicts": [{{"cand_id": "string", "first_3s_hook": 7, "cliffhanger_open": 6, "cold_viewer_clarity": 8,
  "ambiguity": [{{"kind": "relationship", "beat_id": "E01.B010", "detail": "string"}}],
  "ending_resolves_tension": false, "trim_start_to": null, "trim_end_at": null, "trim_reason": "",
  "hook_line": "exact words", "cliffhanger_line": "exact words", "viewer_takeaway": "string", "why_it_works": "string"}}]}}"""


def critique_batch(cands: list[Candidate], idx: Index, client: LLMClient, tag: str) -> list[CriticVerdictLLM]:
    res = client.ask_json(critic_prompt(cands, idx), CriticBatchLLM, tag)
    wanted = {c.cand_id for c in cands}
    return [v for v in res.verdicts if v.cand_id in wanted]


# ------------------------------------------------------------------ applying trims
def _norm_id(x: str | None) -> str:
    return (x or "").strip().upper().replace(" ", "")


def apply_verdict(c: Candidate, v: CriticVerdictLLM | None, idx: Index):
    """Returns (segments, hook_beat, cliff_beat, changes, issues). Trims are applied only if safe."""
    segs = [s.model_copy() for s in c.segments]
    hook, cliff = c.hook_beat, c.cliff_beat
    changes: list[str] = []
    issues: list[Issue] = []
    if v is None:
        return segs, hook, cliff, changes, issues

    def total(ss: list[Segment]) -> float:
        return sum(s.duration for s in ss)

    sid = _norm_id(v.trim_start_to)
    if sid:
        first = segs[0]
        inside = {b.beat_id: b for b in idx.beats_in(first.episode, first.start_sec, first.end_sec)}
        b = inside.get(sid)
        if b is None:
            issues.append(Issue(code="trim_ignored", severity="warn", source="critic",
                                detail=f"trim_start_to {sid} is not inside segment 1"))
        elif b.start_sec > first.start_sec + 0.05:
            new_start = safe_start(idx.beats(first.episode), b.start_sec, 3.0)
            trial = [_seg(idx, first.episode, new_start, first.end_sec, first.role)] + segs[1:] if new_start is not None else None
            if trial and new_start < first.end_sec - 1.0 and MIN_S <= total(trial) <= MAX_S:
                changes.append(f"start {fmt(first.start_sec)} -> {fmt(new_start)} (critic: {v.trim_reason or 'weak opening'})")
                segs, hook = trial, b.beat_id
            else:
                issues.append(Issue(code="trim_discarded", severity="warn", source="code",
                                    detail=f"opening trim to {sid} would make the clip {total(trial) if trial else 0:.1f}s or cut a line"))
    eid = _norm_id(v.trim_end_at)
    if eid:
        last = segs[-1]
        inside = {b.beat_id: b for b in idx.beats_in(last.episode, last.start_sec, last.end_sec)}
        b = inside.get(eid)
        if b is None:
            issues.append(Issue(code="trim_ignored", severity="warn", source="critic",
                                detail=f"trim_end_at {eid} is not inside the last segment"))
        else:
            new_end = safe_end(idx.beats(last.episode), b.end_sec)
            if new_end < last.end_sec - 0.05:
                trial = segs[:-1] + [_seg(idx, last.episode, last.start_sec, new_end, last.role)]
                if new_end > last.start_sec + 1.0 and MIN_S <= total(trial) <= MAX_S:
                    changes.append(f"end {fmt(last.end_sec)} -> {fmt(new_end)} (critic: {v.trim_reason or 'ending resolved'})")
                    segs, cliff = trial, b.beat_id
                else:
                    issues.append(Issue(code="trim_discarded", severity="warn", source="code",
                                        detail=f"ending trim to {eid} would make the clip {total(trial):.1f}s"))
    return segs, hook, cliff, changes, issues


# ------------------------------------------------------------------ tiers + schedule
def _score(x: float) -> int:
    return int(round(min(10, max(1, x))))


def classify(p: Promo) -> tuple[str, list[str]]:
    if p.blocking:
        return "REJECTED", [i.detail for i in p.blocking]
    v = p.verdict
    if v is None:
        return "UNREVIEWED", ["critic has not evaluated this candidate"]
    hook, cliff, clar = _score(v.first_3s_hook), _score(v.cliffhanger_open), _score(v.cold_viewer_clarity)
    kinds = {a.kind.strip().lower().replace(" ", "_") for a in v.ambiguity}
    review = []
    if clar <= 5:
        review.append(f"low cold-viewer clarity ({clar}/10)")
    if kinds & SEVERE_AMBIGUITY and clar <= AMBIGUITY_FLAG_CLARITY:
        review.append(f"ambiguity at clarity {clar}/10: " + ", ".join(sorted(kinds & SEVERE_AMBIGUITY)))
    if v.ending_resolves_tension:
        review.append("ending resolves the tension")
    if cliff <= 4:
        review.append(f"weak cliffhanger ({cliff}/10)")
    if hook <= 4:
        review.append(f"weak first 3 seconds ({hook}/10)")
    if review:
        return "TIER_3", review
    if hook >= 8 and clar >= 8 and cliff >= 7 and not kinds & TIER1_BLOCKERS:
        return "TIER_1", [f"hook {hook}/10 in the first 3 s", f"clarity {clar}/10, no prior context needed",
                          f"cliffhanger {cliff}/10"]
    why = [f"hook {hook}/10, cliffhanger {cliff}/10, clarity {clar}/10"]
    if kinds:
        why.append("minor ambiguity: " + ", ".join(sorted(kinds)))
    return "TIER_2", why


def _key(p: Promo) -> tuple:
    v = p.verdict
    h, c, k = _score(v.first_3s_hook), _score(v.cliffhanger_open), _score(v.cold_viewer_clarity)
    if p.tier == "TIER_1":
        return (-h, -k, -c, -p.hook_intensity, p.promo_id)
    if p.tier == "TIER_2":
        return (-c, -h, -k, -p.cliffhanger_tension, p.promo_id)
    return (-(c + h), -k, p.promo_id)


def _space_out(items: list[Promo]) -> list[Promo]:
    """Keep sort order, but avoid posting the same episode twice in a row when there is a choice."""
    rest, out = list(items), []
    while rest:
        pick = next((p for p in rest if not out or set(p.episodes).isdisjoint(out[-1].episodes)), rest[0])
        rest.remove(pick)
        out.append(pick)
    return out


def schedule(promos: list[Promo]) -> list[Promo]:
    for p in promos:
        p.tier, p.tier_reasons = classify(p)
        p.post_slot = None
    order = {"TIER_1": 0, "TIER_2": 1, "TIER_3": 2, "UNREVIEWED": 3, "REJECTED": 4}
    out: list[Promo] = []
    slot = 1
    for tier in ("TIER_1", "TIER_2", "TIER_3"):
        group = _space_out(sorted([p for p in promos if p.tier == tier], key=_key))
        for p in group:
            if tier != "TIER_3":     # flagged promos are not scheduled until a human clears them
                p.post_slot = slot
                slot += 1
        out += group
    out += sorted([p for p in promos if order[p.tier] > 2], key=lambda p: (order[p.tier], p.promo_id))
    return out
