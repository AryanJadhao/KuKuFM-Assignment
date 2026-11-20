"""LLM step 1 - beat-level anchor detection.

The model does NOT assemble promos. It reads one episode as a numbered list of beats and marks
single beats of two atomic kinds:

  HOOK_ANCHOR         a beat that works as the first thing a scrolling stranger sees (hook_intensity 1-10)
  CLIFFHANGER_ANCHOR  a beat where tension peaks and is left hanging (cliffhanger_tension 1-10)

Everything the model returns is checked in code before it is used:
  * the beat_id must exist in this episode
  * the excerpt must be grounded in that beat's text; if it is grounded in a beat within +-3 positions
    instead, the anchor is moved there (logged as `repointed_from`); otherwise the anchor is dropped
  * strengths are clamped to 1-10, duplicates on the same beat keep the strongest
Dropped anchors are kept in the per-episode output file with the reason, for the quality report.
"""
from __future__ import annotations

from .llm_client import LLMClient
from .models import Anchor, AnchorSheetLLM, Beat, EpisodeScript
from .parser import fmt
from .text_match import GROUNDED, grounded_ratio

PROMPT_VERSION = "anchors-v1"


def beat_line(b: Beat) -> str:
    who = ""
    if b.type == "DIALOGUE":
        who = f" {b.speaker}" + (f" ({b.direction})" if b.direction else "")
    return f"{b.beat_id} [{fmt(b.start_sec)}-{fmt(b.end_sec)}] {b.type}{who}: {b.text}"


def anchor_prompt(ep: EpisodeScript, max_hooks: int, max_cliffs: int, beats: list[Beat] | None = None,
                  part_note: str = "") -> str:
    beats = beats if beats is not None else ep.beats
    listing = "\n".join(beat_line(b) for b in beats)
    return f"""ROLE
You scout screenplay beats for a vertical-video ad team. You do NOT build promos and you do NOT pick
start or end times. You only mark individual beats that could work as an opening jolt or as a closing sting.
Code will later combine your marks into 20-60 second clips, so mark the single best beat of each moment.

EPISODE {ep.episode}
Logline (background for you only): {ep.logline}
{part_note}
BEATS (one per line: BEAT_ID [start-end] TYPE SPEAKER (direction): text)
CAMERA beats describe the picture and overlap the dialogue in time.
{listing}

WHAT TO MARK
HOOK_ANCHOR - a beat that would stop a stranger's thumb if it were the first 3 seconds of a clip:
  a physical shock (slap, shove, grab, collapse, weapon), a line that lands as an accusation, insult,
  threat or confession, or a sudden reveal of identity or betrayal. Scenery, establishing shots, entrances
  without conflict and polite talk are never hooks.
  hook_intensity: 10 = impossible to scroll past, 7 = strong, 4 = mild curiosity. Do not list anything below 4.
CLIFFHANGER_ANCHOR - a beat where tension peaks and is left open: a secret about to come out, an
  ultimatum, a threat, an interrupted action, someone unexpected appearing, a question nobody answers yet.
  If the beats that follow quickly answer or defuse it, give it a lower cliffhanger_tension.
  cliffhanger_tension: 10 = unbearable not to know what happens next. Do not list anything below 4.

RULES
- Use beat_ids exactly as printed above. Never invent beats, lines or events.
- excerpt: copy 4-20 consecutive words VERBATIM from that beat's text.
- Prefer DIALOGUE and ACTION beats. Mark a CAMERA beat only when the jolt is purely visual.
- At most {max_hooks} hook anchors and {max_cliffs} cliffhanger anchors. Fewer is fine. Spread them over
  different moments of the episode instead of marking neighbouring beats of the same moment.
- device (hooks): one or two words, e.g. slap, reveal, insult, threat, confession, humiliation, collapse.
- open_question (cliffhangers): the question the viewer is left with, in under 15 words.

Return ONLY this JSON:
{{"hook_anchors": [{{"beat_id": "E01.B012", "hook_intensity": 8, "device": "insult", "excerpt": "verbatim words", "why": "short reason"}}],
  "cliffhanger_anchors": [{{"beat_id": "E01.B040", "cliffhanger_tension": 7, "open_question": "string", "excerpt": "verbatim words", "why": "short reason"}}]}}"""


def _chunks(ep: EpisodeScript, max_chars: int = 30000) -> list[list[Beat]]:
    """Long episodes are split on scene boundaries so no prompt gets huge."""
    groups: list[list[Beat]] = [[]]
    size = 0
    for b in ep.beats:
        n = len(beat_line(b)) + 1
        new_scene = groups[-1] and groups[-1][-1].scene != b.scene
        if groups[-1] and size + n > max_chars and new_scene:
            groups.append([])
            size = 0
        groups[-1].append(b)
        size += n
    return [g for g in groups if g]


def validate_sheet(ep: EpisodeScript, sheet: AnchorSheetLLM) -> tuple[list[Anchor], list[dict]]:
    by_id = {b.beat_id: b for b in ep.beats}
    accepted: dict[tuple[str, str], Anchor] = {}
    dropped: list[dict] = []
    items = [("HOOK_ANCHOR", a.beat_id, a.hook_intensity, a.device, a.excerpt, a.why) for a in sheet.hook_anchors] + \
            [("CLIFFHANGER_ANCHOR", a.beat_id, a.cliffhanger_tension, a.open_question, a.excerpt, a.why)
             for a in sheet.cliffhanger_anchors]
    for kind, bid, strength, label, excerpt, why in items:
        bid = (bid or "").strip().upper().replace(" ", "")
        if bid not in by_id:
            dropped.append({"kind": kind, "beat_id": bid, "reason": "beat_id not in this episode", "excerpt": excerpt})
            continue
        beat = by_id[bid]
        repointed = None
        if grounded_ratio(excerpt, beat.text) < GROUNDED:
            near = [b for b in ep.beats if 0 < abs(b.order - beat.order) <= 3]
            best = max(near, key=lambda b: grounded_ratio(excerpt, b.text), default=None)
            if best is not None and grounded_ratio(excerpt, best.text) >= GROUNDED:
                repointed, beat = bid, best
            else:
                dropped.append({"kind": kind, "beat_id": bid, "reason": "excerpt not found in the screenplay text",
                                "excerpt": excerpt})
                continue
        a = Anchor(kind=kind, beat_id=beat.beat_id, episode=ep.episode, strength=int(round(min(10, max(1, strength)))),
                   label=label.strip(), excerpt=excerpt.strip(), why=why.strip(), repointed_from=repointed)
        key = (kind, beat.beat_id)
        if key not in accepted or accepted[key].strength < a.strength:
            accepted[key] = a
    order = {b.beat_id: b.order for b in ep.beats}
    return sorted(accepted.values(), key=lambda a: (a.kind, order[a.beat_id])), dropped


def tag_episode(ep: EpisodeScript, client: LLMClient, max_hooks: int = 6, max_cliffs: int = 6) -> dict:
    """Returns the per-episode record saved to anchors/ep_XXX.json."""
    groups = _chunks(ep)
    merged = AnchorSheetLLM()
    models = []
    for gi, beats in enumerate(groups, 1):
        note = "" if len(groups) == 1 else f"(Long episode: part {gi} of {len(groups)}.)\n"
        tag = f"anchors_ep{ep.episode:03d}" + (f"_part{gi}" if len(groups) > 1 else "")
        sheet = client.ask_json(anchor_prompt(ep, max_hooks, max_cliffs, beats, note), AnchorSheetLLM, tag)
        merged.hook_anchors += sheet.hook_anchors
        merged.cliffhanger_anchors += sheet.cliffhanger_anchors
        if client.last_model not in models:
            models.append(client.last_model)
    accepted, dropped = validate_sheet(ep, merged)
    return {"episode": ep.episode, "prompt_version": PROMPT_VERSION, "model": " + ".join(models),
            "model_output": merged.model_dump(), "accepted": [a.model_dump() for a in accepted], "dropped": dropped}
