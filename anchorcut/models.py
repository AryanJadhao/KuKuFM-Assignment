"""Data shapes shared by every stage.

Three groups:
  * Screenplay  - Beat / SceneInfo / EpisodeScript / ParseIssue      (built by parser.py, no LLM)
  * Anchors     - what the anchor tagger LLM returns + validated Anchor
  * Promos      - Window candidates (pure arithmetic) and the final, critic-reviewed Promo

Rule that runs through the whole project: the LLM only ever names beat_ids.
Every second on a Promo is computed in Python from the parsed screenplay.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BeatType = Literal["CAMERA", "ACTION", "SFX", "DIALOGUE"]


# --------------------------------------------------------------------------- screenplay
class Beat(BaseModel):
    beat_id: str                 # "E07.B031" -> episode 7, 31st beat (0-based) of that episode
    episode: int
    scene: int
    order: int
    start_sec: float
    end_sec: float
    type: BeatType
    speaker: Optional[str] = None
    direction: Optional[str] = None   # parenthetical on a speaker tag, e.g. "Cold, Commanding"
    text: str
    raw_start: str                    # timestamp exactly as written in the script
    raw_end: str

    @property
    def length(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


class SceneInfo(BaseModel):
    number: int
    heading: str = ""
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


class ParseIssue(BaseModel):
    file: str
    line_no: int
    line: str
    reason: str


class EpisodeScript(BaseModel):
    episode: int
    file: str
    logline: str = ""
    continuity: str = ""
    scenes: list[SceneInfo] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    duration_sec: float = 0.0
    issues: list[ParseIssue] = Field(default_factory=list)


# --------------------------------------------------------------------------- LLM step 1: anchors
class HookAnchorLLM(BaseModel):
    beat_id: str
    hook_intensity: float
    device: str = ""
    excerpt: str
    why: str = ""


class CliffAnchorLLM(BaseModel):
    beat_id: str
    cliffhanger_tension: float
    open_question: str = ""
    excerpt: str
    why: str = ""


class AnchorSheetLLM(BaseModel):
    hook_anchors: list[HookAnchorLLM] = Field(default_factory=list)
    cliffhanger_anchors: list[CliffAnchorLLM] = Field(default_factory=list)


class Anchor(BaseModel):
    """A validated anchor: beat exists, excerpt really appears in that beat's text."""
    kind: Literal["HOOK_ANCHOR", "CLIFFHANGER_ANCHOR"]
    beat_id: str
    episode: int
    strength: int                    # hook_intensity or cliffhanger_tension, clamped 1-10
    label: str = ""                  # device (hook) or open question (cliffhanger)
    excerpt: str
    why: str = ""
    repointed_from: Optional[str] = None   # the model named a neighbouring beat; we moved it

    @property
    def hook_intensity(self) -> Optional[int]:
        return self.strength if self.kind == "HOOK_ANCHOR" else None

    @property
    def cliffhanger_tension(self) -> Optional[int]:
        return self.strength if self.kind == "CLIFFHANGER_ANCHOR" else None


# --------------------------------------------------------------------------- windows (pure python)
class Segment(BaseModel):
    episode: int
    start_sec: float
    end_sec: float
    first_beat: str
    last_beat: str
    role: Literal["arc", "flash_forward", "scene_body"]

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


class Candidate(BaseModel):
    cand_id: str
    mode: Literal["A", "B"]          # A = continuous arc, B = flash-forward cold open + scene stitch
    segments: list[Segment]
    hook_beat: str
    cliff_beat: str
    hook_intensity: int
    cliffhanger_tension: int
    hook_offset: float               # seconds from clip start to the hook beat (<= 3 by construction)
    prior: float                     # structural pre-score used only to choose what the critic sees
    notes: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out


# --------------------------------------------------------------------------- LLM step 2: critic
class Ambiguity(BaseModel):
    kind: str                        # pronoun | relationship | unexplained_name | prior_event | speaker_identity
    beat_id: str = ""
    detail: str = ""


class CriticVerdictLLM(BaseModel):
    cand_id: str
    first_3s_hook: float
    cliffhanger_open: float
    cold_viewer_clarity: float
    ambiguity: list[Ambiguity] = Field(default_factory=list)
    ending_resolves_tension: bool = False
    trim_start_to: Optional[str] = None
    trim_end_at: Optional[str] = None
    trim_reason: str = ""
    hook_line: str = ""
    cliffhanger_line: str = ""
    viewer_takeaway: str = ""
    why_it_works: str = ""


class CriticBatchLLM(BaseModel):
    verdicts: list[CriticVerdictLLM]


# --------------------------------------------------------------------------- final promo
Tier = Literal["TIER_1", "TIER_2", "TIER_3", "REJECTED", "UNREVIEWED"]
TIER_LABEL = {
    "TIER_1": "Tier 1 · Cold-traffic acquisition",
    "TIER_2": "Tier 2 · High-stakes escalation",
    "TIER_3": "Tier 3 · Human review flagged",
    "REJECTED": "Rejected by verification",
    "UNREVIEWED": "Not yet reviewed by the critic",
}


class Issue(BaseModel):
    code: str
    severity: Literal["block", "warn"]
    detail: str
    source: Literal["code", "critic"]   # code = verified against the screenplay; critic = model opinion


class Promo(BaseModel):
    promo_id: str
    mode: Literal["A", "B"]
    segments: list[Segment]
    original_segments: list[Segment]
    hook_beat: str
    cliff_beat: str
    hook_intensity: int
    cliffhanger_tension: int
    hook_offset: float
    duration: float
    hook_text: str
    cliff_text: str
    transcript: list[str] = Field(default_factory=list)
    verdict: Optional[CriticVerdictLLM] = None
    boundary_changes: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    tier: Tier = "UNREVIEWED"
    tier_reasons: list[str] = Field(default_factory=list)
    post_slot: Optional[int] = None
    anchor_model: str = ""
    critic_model: str = ""

    @property
    def episodes(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "block"]
