# AnchorCut: promo finder for microdrama screenplays

AnchorCut was built for Kuku AI Generalist Assignment Problem 1, "Find the Promos". It reads a ZIP of timestamped episode screenplays and returns 20–60 s promos. Each promo comes with:

- the episode and the start/end timestamps of every segment, in play order
- the hook and the cliffhanger
- one line on why it works
- a **campaign tier** that says what to post first

> Screenplays only. No video is cut or played; every timestamp refers to the screenplay timeline.

## Approach: Anchor → Expand → Verify

```
ZIP ─► parser.py ─────────────► beats (beat_id, episode, start_sec, end_sec, type, speaker, text) + parse issues
            │                                                                           (pure Python)
            ├─► anchor_tagger.py ──► LLM step 1, one call per episode: tag single beats as
            │                         HOOK_ANCHOR (hook_intensity 1-10) / CLIFFHANGER_ANCHOR (cliffhanger_tension 1-10)
            │                         → code grounds every excerpt in the beat text, drops invented ones
            ├─► window_builder.py ─► pure arithmetic: Mode A continuous arc  [hook … cliffhanger], 20-60 s
            │                         Mode B flash-forward cold open (3-8 s) + scene ending on a cliffhanger
            │                         → no mid-line cuts, hook ≤ 3 s in, dedup, ≤ N per episode
            ├─► cold_viewer_critic.py ► LLM step 2, batched: a first-time viewer scores first-3s hook,
            │                         cliffhanger, cold-viewer clarity, flags ambiguity, may trim boundaries
            ├─► verify.py ───────────► code re-checks every boundary against the screenplay
            └─► tiers + schedule ────► Tier 1 cold traffic · Tier 2 escalation · Tier 3 human review
```

| Decision | Why |
|---|---|
| The LLM marks **single beats**, not whole promos | Picking an exciting line is easy for a model. Arithmetic over 3,700 timestamps is not. So code chooses the boundaries and the model never writes a timestamp. |
| Window building is pure Python | The 20–60 s rule, the 3-second hook rule and "never cut a line in half" are exact rules, so code enforces them exactly. |
| The critic never sees why a clip was built | It can't be anchored by the tagger's enthusiasm. It reads only what a stranger would see: the clip text with `t = seconds into the clip`, and speaker names marked as production notes. |
| Trims are applied only if they stay safe | A trim has to land on a beat inside the clip and keep the clip at 20–60 s. Anything else is discarded and logged. |
| Tiers come from readable rules, not a weighted sum | "Why is this #1?" has a one-line answer, and low-clarity clips are **flagged** for a human instead of being quietly ranked low. |
| Every model output is checked | Excerpts and quotes are matched against the screenplay text (`text_match.py`). Issues are labelled `code` (verified) or `critic` (model opinion). |

### Tier rules (`cold_viewer_critic.py`)

| Tier | Rule | Posting order within the tier |
|---|---|---|
| **Tier 1 · Cold-traffic acquisition** | hook ≥ 8, clarity ≥ 8, cliffhanger ≥ 7, and no relationship or prior-event confusion flagged | hook, then clarity, then cliffhanger |
| **Tier 2 · High-stakes escalation** | Passed verification, but doesn't meet the Tier 1 or Tier 3 rules | cliffhanger, then hook, then clarity |
| **Tier 3 · Human review flagged** | Any of: clarity ≤ 5; a relationship, prior-event or speaker-identity ambiguity together with clarity ≤ 6; the ending resolves the tension; cliffhanger ≤ 4; hook ≤ 4 | Not scheduled until an editor clears it |

**Calibration change after the first full run.** The first version sent every clip with *any* relationship or prior-event note to Tier 3. The critic adds such a note to almost every clip from the middle of a series, even when it rates clarity 7–8/10, so 81% of the first 85 reviewed clips were flagged and the review queue was meaningless. Now an ambiguity note flags a clip only when clarity is ≤ 6. On those 85 clips that changes the split from 2 / 14 / 69 to 9 / 48 / 28 (Tier 1 / 2 / 3). The new threshold was chosen on the same run, with no held-out data, so treat it as a starting point for an editor to calibrate.
| Rejected | Failed a code check: duration, hook position, mid-line cut, overlap or bounds | — |

Within the schedule, two posts in a row never come from the same episode when another option exists.

## Assumptions about timestamps and duration

1. Times are `MM:SS.cc` from the start of each episode. `HH:MM:SS.cc` is also accepted. Every episode starts at 00:00.
2. A beat is one tagged block in the script: `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**`, or `**SPEAKER (direction):**`, followed by a `Start: (…) | End: (…)` line and its text. CAMERA beats overlap dialogue in time; that is expected, not an error.
3. Where a segment plays: a segment `[start, end]` plays every beat that **begins** inside it.
4. Duration rule: **promo duration = Σ (segment end − segment start)**, and must be **20.0 ≤ duration ≤ 60.0 s**. Hard cuts between segments take no time.
5. Hook rule: the hook beat must **begin within 3.0 s** of the clip start. Mode A starts exactly on the hook beat; it only moves earlier, by at most 3 s, to avoid starting in the middle of a line.
6. **Never cut a spoken line.** A segment can't start or end while a DIALOGUE beat is in progress. The end also waits for an ACTION beat in progress to finish. CAMERA and SFX beats may be cut.
7. A cliffhanger should be the final beat. If the clip runs more than 2.5 s past the cliffhanger beat, it gets a warning.
8. Mode B (flash-forward) segments:
   - The cold open is 3–8 s and comes from **later** in the story than the scene that follows it: the same episode or the next one.
   - The scene body ends on a cliffhanger. Where possible it stays inside that cliffhanger's scene, and the clip aims for about 40 s in total.
9. Malformed or missing timestamps are **logged**, never silently dropped:
   - A beat without a usable timestamp is skipped.
   - A line outside any beat is ignored.
   - Both are listed in `parsed/parse_issues.json` and in the Quality view.
   - On the supplied pack, all 3,727 timing lines parse with 0 issues.
10. The episode number comes from the `EPISODE NUMBER:` header, or from the filename if the header is missing (logged).

## Run it

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # put your key in LLM_API_KEY
python -m pytest                # offline tests: synthetic screenplay + a scripted fake model, no API calls
streamlit run app.py            # UI
```

For the full 50-episode pack, the CLI is the most robust option:

```powershell
python -m anchorcut.cli run "path\to\pack.zip" --label duke
python -m anchorcut.cli resume <run_id>      # after a quota pause or a crash: finished work is skipped
python -m anchorcut.cli rebuild <run_id>     # re-verify and re-tier from saved outputs, no LLM calls
python -m anchorcut.cli parse "path\to\pack.zip"   # parser only, prints every parse issue
```

Cost is about one call per episode plus one call per 5 candidates. For 50 episodes at 3 candidates each, that is about 50 + 30 = 80 calls, which fits a free Gemini tier.

## Swapping the model and API key (no code changes)

There are three ways. When more than one sets the same thing, the first one wins:

1. **In the app:** sidebar → **Model & API key**. Pick the provider, type a model name and optionally paste a key, then **Apply**. **Test connection** sends one tiny request to check the setup. This applies to runs you start from that browser session only. Nothing is written to disk, and the key is never displayed, logged or saved in outputs.
2. **Streamlit Cloud Secrets, or real environment variables.**
3. **The `.env` file:** it is re-read on every page load, so a new key or model applies without restarting the app.

| Provider (`LLM_PROVIDER`) | Key variable | Default model | Endpoint |
|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` | gemini-3.5-flash | Google AI Studio |
| `openrouter` | `OPENROUTER_API_KEY` | google/gemini-2.5-flash | openrouter.ai |
| `groq` | `GROQ_API_KEY` | openai/gpt-oss-120b | api.groq.com |
| `openai` | `OPENAI_API_KEY` | (set `LLM_MODEL`) | `LLM_BASE_URL`: **any OpenAI-compatible gateway** |

- `LLM_API_KEY`, if set, is used for whichever provider is selected. Otherwise each provider uses only its own key variable, so switching provider never sends one company's key to another.
- `LLM_FALLBACK_MODELS` (same provider) takes over when a model's daily quota runs out. If none is left, the run pauses and `resume` continues later.
- A run in progress keeps the model it started with. Saved outputs record which model produced each one, and the report warns when a run mixed models.

## The UI (`app.py`): Editorial Storyboard & Timeline Inspector

The app uses a light theme and has five views:

- **Posting Schedule**: one tab per tier (Tier 1, Tier 2, Tier 3 flagged, Rejected, Unreviewed). Each storyboard card shows:
  - the clip on a fixed 0–60 s ruler: amber for the first 3 s, slate for the body, rust for the cliffhanger beat, and black lines for cuts
  - the hook and cliffhanger lines, why it works, the segments, the scores and the tier reasoning
- **Timeline Inspector**: the selected promo's beats in lanes (shot, action, line, sound) against seconds into the promo, plus:
  - the critic's ambiguity flags and the boundaries before and after any trim
  - every check result
  - the clip transcript as it plays
- **Episode Anchor Map**: every beat of an episode, the anchors plotted by strength, the promo windows coloured by tier, the anchors that were dropped, and the full screenplay reader.
- **Quality**: tier and mode chart, the quality report, and parse issues.
- **Export**: one-click download of `raw_outputs/` (ZIP), plus the posting schedule CSV, promos JSON and quality report.

Changing filters or views never calls the model. Processing runs in a background thread and reports through `status.json`.

## Outputs, per run (`workspace/runs/<run_id>/`)

| Path | Contents |
|---|---|
| `raw_outputs/*.json` | Every LLM call, **unedited**: prompt, raw response text, model, time and parse error, one file per attempt |
| `anchors/ep_XXX.json` | The model's anchor sheet, plus the accepted anchors and the dropped ones (with the reason each was dropped) |
| `candidates.json` | Windows built by code, plus selection stats |
| `critic/<id>.json` | Critic verdict per candidate |
| `results/posting_schedule.csv`, `results/promos.json` | Final ranked promos |
| `results/quality_report.md` | Coverage, tiers, and where it picks badly |
| `parsed/`, `status.json`, `run.log` | Parse output, progress and errors |

The uploaded ZIP is never stored or exported. `workspace/` is gitignored because the files in it quote the confidential screenplays.

## Deploy (Streamlit Community Cloud)

Main file: `app.py`. Put `LLM_API_KEY`, `LLM_PROVIDER` and `LLM_MODEL` in the app's **Secrets**, never in the repo. A public deployment should not be used with the confidential pack. Keep it private, or demo it with your own screenplay.

## Known limits

- **A screenplay is not footage.** Acting, music and pacing decide whether a clip really works. Only the hook's position is a guarantee; whether it actually stops a scroll is the critic's opinion.
- **The critic is an LLM judging LLM picks, with no ground truth.** Tier 3 and the Quality view show where it is unsure, but an editor's accept/reject decisions are needed before trusting the ranking.
- **Speaker labels in the source are inconsistent** (the same character appears under two different tags, and some characters are only "a nobleman"). The critic is told that names are production notes, but the parser does not merge them.
- **Mode B only pairs adjacent episodes**, and anchors are capped per episode. A strong moment the tagger doesn't mark is never considered (a false negative).
