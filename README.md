<p align="center">
  <h1 align="center">🎬 AnchorCut</h1>
  <p align="center"><strong>Promo Finder for Microdrama Screenplays</strong></p>
  <p align="center">
    <em>Built for Kuku AI Generalist Assignment — Problem 1: "Find the Promos"</em>
  </p>
</p>

---

AnchorCut reads a ZIP of timestamped episode screenplays and returns **20–60 s promos**. Each promo comes with:

- 📍 The episode and the **start/end timestamps** of every segment, in play order
- 🪝 The **hook** and the **cliffhanger**
- 💡 One line on **why it works**
- 📊 A **campaign tier** that says what to post first

> [!NOTE]
> Screenplays only. No video is cut or played; every timestamp refers to the screenplay timeline.

---

## 🧠 Approach: Anchor → Expand → Verify

```
ZIP ─► parser.py ──────────────► beats (beat_id, episode, start_sec, end_sec, type, speaker, text)
            │                     + parse issues (pure Python)
            │
            ├─► anchor_tagger.py ──► LLM step 1 (one call per episode):
            │                         tag single beats as HOOK_ANCHOR / CLIFFHANGER_ANCHOR
            │                         → code grounds every excerpt, drops invented ones
            │
            ├─► window_builder.py ─► Pure arithmetic:
            │                         Mode A — continuous arc [hook … cliffhanger], 20-60 s
            │                         Mode B — flash-forward cold open (3-8 s) + cliffhanger scene
            │                         → no mid-line cuts, hook ≤ 3 s in, dedup, ≤ N per episode
            │
            ├─► cold_viewer_critic.py ► LLM step 2 (batched):
            │                         first-time viewer scores hook, cliffhanger, clarity
            │                         flags ambiguity, may trim boundaries
            │
            ├─► verify.py ──────────► Code re-checks every boundary against the screenplay
            │
            └─► tiers + schedule ───► Tier 1 · Tier 2 · Tier 3 · Rejected
```

### 🔑 Key Design Decisions

| Decision | Why |
|---|---|
| LLM marks **single beats**, not whole promos | Picking an exciting line is easy for a model. Arithmetic over 3,700 timestamps is not. Code chooses boundaries — the model never writes a timestamp. |
| Window building is **pure Python** | The 20–60 s rule, 3-second hook rule, and "never cut a line in half" are exact rules — code enforces them exactly. |
| Critic **never sees** why a clip was built | It can't be anchored by the tagger's enthusiasm. It reads only what a stranger would see. |
| Trims applied only if **safe** | A trim must land on a beat inside the clip and keep 20–60 s. Anything else is discarded and logged. |
| Tiers use **readable rules**, not weighted sums | "Why is this #1?" has a one-line answer. Low-clarity clips are **flagged** for a human instead of quietly ranked low. |
| Every model output is **verified** | Excerpts and quotes matched against screenplay text (`text_match.py`). Issues labelled `code` (verified) or `critic` (model opinion). |

---

## 🏷️ Tier Rules

> Defined in `cold_viewer_critic.py`

| Tier | Rule | Posting Order |
|:---:|---|---|
| **🟢 Tier 1** | hook ≥ 8, clarity ≥ 8, cliffhanger ≥ 7, no relationship/prior-event confusion | hook → clarity → cliffhanger |
| **🟡 Tier 2** | Passed verification, doesn't meet Tier 1 or Tier 3 rules | cliffhanger → hook → clarity |
| **🔴 Tier 3** | clarity ≤ 5; ambiguity + clarity ≤ 6; ending resolves tension; cliffhanger ≤ 4; hook ≤ 4 | Not scheduled until editor clears it |
| **⛔ Rejected** | Failed code check: duration, hook position, mid-line cut, overlap, or bounds | — |

> [!IMPORTANT]
> **Calibration note:** The first version flagged every clip with *any* relationship/prior-event note → 81% ended up in Tier 3. Now ambiguity flags only when clarity ≤ 6, changing the split from 2/14/69 to **9/48/28** (Tier 1/2/3). This threshold was chosen on the same run with no held-out data — treat it as a starting point.

Within the schedule, two posts in a row never come from the same episode when another option exists.

---

## 📐 Assumptions about Timestamps & Duration

<details>
<summary><strong>Click to expand all assumptions</strong></summary>

1. Times are `MM:SS.cc` from the start of each episode. `HH:MM:SS.cc` is also accepted. Every episode starts at `00:00`.
2. A beat is one tagged block in the script: `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**`, or `**SPEAKER (direction):**`, followed by a `Start: (…) | End: (…)` line and its text. CAMERA beats overlap dialogue in time — that is expected, not an error.
3. Where a segment plays: a segment `[start, end]` plays every beat that **begins** inside it.
4. **Duration rule:** promo duration = Σ (segment end − segment start), must be **20.0 ≤ duration ≤ 60.0 s**. Hard cuts between segments take no time.
5. **Hook rule:** the hook beat must **begin within 3.0 s** of the clip start. Mode A starts exactly on the hook beat; it only moves earlier by at most 3 s to avoid starting mid-line.
6. **Never cut a spoken line.** A segment can't start or end while a DIALOGUE beat is in progress. The end also waits for an ACTION beat to finish. CAMERA and SFX beats may be cut.
7. A cliffhanger should be the final beat. If the clip runs more than 2.5 s past the cliffhanger beat, it gets a warning.
8. **Mode B (flash-forward) segments:**
   - Cold open is 3–8 s, from **later** in the story than the scene that follows it (same episode or next).
   - Scene body ends on a cliffhanger. Where possible it stays inside that cliffhanger's scene, aiming for ~40 s total.
9. **Malformed or missing timestamps** are **logged**, never silently dropped:
   - Beat without a usable timestamp → skipped
   - Line outside any beat → ignored
   - Both listed in `parsed/parse_issues.json` and in the Quality view
   - On the supplied pack, all 3,727 timing lines parse with **0 issues**
10. Episode number comes from the `EPISODE NUMBER:` header, or from the filename if the header is missing (logged).

</details>

---

## 🚀 Quick Start

```powershell
# 1. Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure API key
copy .env.example .env          # put your key in LLM_API_KEY

# 3. Run tests (offline — no API calls needed)
python -m pytest

# 4. Launch the UI
streamlit run app.py
```

### CLI Commands (for full 50-episode pack)

```powershell
python -m anchorcut.cli run "path\to\pack.zip" --label duke
python -m anchorcut.cli resume <run_id>       # resume after quota pause or crash
python -m anchorcut.cli rebuild <run_id>      # re-verify & re-tier, no LLM calls
python -m anchorcut.cli parse "path\to\pack.zip"   # parser only, prints parse issues
```

> **💰 Cost:** ~1 call per episode + 1 call per 5 candidates. For 50 episodes at 3 candidates each ≈ **80 calls** — fits a free Gemini tier.

---

## 🔧 Swapping the Model & API Key

> No code changes required — three ways to configure (first one wins):

| Method | Details |
|---|---|
| **🖥️ In the app** | Sidebar → *Model & API key* → Pick provider, type model name, paste key → **Apply**. Use **Test connection** to verify. Session-only; nothing written to disk. |
| **☁️ Streamlit Cloud Secrets** | Or real environment variables |
| **📄 `.env` file** | Re-read on every page load — new key/model applies without restart |

### Supported Providers

| Provider (`LLM_PROVIDER`) | Key Variable | Default Model | Endpoint |
|:---:|---|---|---|
| `gemini` | `GEMINI_API_KEY` | gemini-3.5-flash | Google AI Studio |
| `openrouter` | `OPENROUTER_API_KEY` | google/gemini-2.5-flash | openrouter.ai |
| `groq` | `GROQ_API_KEY` | openai/gpt-oss-120b | api.groq.com |
| `openai` | `OPENAI_API_KEY` | *(set `LLM_MODEL`)* | `LLM_BASE_URL` — any OpenAI-compatible gateway |

> [!TIP]
> - `LLM_API_KEY`, if set, is used for whichever provider is selected. Otherwise each provider uses only its own key variable — switching provider never sends one company's key to another.
> - `LLM_FALLBACK_MODELS` (same provider) takes over when a model's daily quota runs out. If none is left, the run pauses and `resume` continues later.
> - A run in progress keeps the model it started with. Saved outputs record which model produced each one, and the report warns when a run mixed models.

---

## 🖼️ The UI — Editorial Storyboard & Timeline Inspector

> **Main file:** `app.py` · Light theme · Five views

| View | What It Shows |
|---|---|
| **📋 Posting Schedule** | One tab per tier. Each storyboard card shows the clip on a fixed 0–60 s ruler (amber = first 3 s, slate = body, rust = cliffhanger, black lines = cuts), hook/cliffhanger lines, scores, and tier reasoning |
| **⏱️ Timeline Inspector** | Selected promo's beats in lanes (shot, action, line, sound) against seconds. Includes critic's ambiguity flags, boundaries before/after trim, check results, and clip transcript |
| **🗺️ Episode Anchor Map** | Every beat of an episode, anchors plotted by strength, promo windows coloured by tier, dropped anchors, and full screenplay reader |
| **📊 Quality** | Tier and mode chart, quality report, and parse issues |
| **📦 Export** | One-click download of `raw_outputs/` (ZIP), posting schedule CSV, promos JSON, and quality report |

Changing filters or views **never calls the model**. Processing runs in a background thread and reports through `status.json`.

---

## 📁 Output Structure

> Per run: `workspace/runs/<run_id>/`

| Path | Contents |
|---|---|
| `raw_outputs/*.json` | Every LLM call **unedited**: prompt, raw response, model, time, parse error — one file per attempt |
| `anchors/ep_XXX.json` | Model's anchor sheet + accepted/dropped anchors with reasons |
| `candidates.json` | Windows built by code + selection stats |
| `critic/<id>.json` | Critic verdict per candidate |
| `results/posting_schedule.csv` | Final ranked promos (CSV) |
| `results/promos.json` | Final ranked promos (JSON) |
| `results/quality_report.md` | Coverage, tiers, and where it picks badly |
| `parsed/`, `status.json`, `run.log` | Parse output, progress, and errors |

> [!NOTE]
> The uploaded ZIP is never stored or exported. `workspace/` is gitignored because the files in it quote the confidential screenplays.

---

## ☁️ Deploy (Streamlit Community Cloud)

**Main file:** `app.py`

Put `LLM_API_KEY`, `LLM_PROVIDER`, and `LLM_MODEL` in the app's **Secrets** — never in the repo.

> [!CAUTION]
> A public deployment should not be used with the confidential pack. Keep it private, or demo it with your own screenplay.

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| 🎥 **Screenplay ≠ footage** | Acting, music, and pacing decide whether a clip really works. Only the hook's position is guaranteed; whether it stops a scroll is the critic's opinion. |
| 🤖 **LLM judging LLM picks** | No ground truth. Tier 3 and the Quality view show uncertainty, but an editor's accept/reject decisions are needed before trusting the ranking. |
| 🏷️ **Inconsistent speaker labels** | Same character appears under different tags; some are only "a nobleman". The critic is told names are production notes, but the parser does not merge them. |
| 🔗 **Mode B pairs adjacent episodes only** | Anchors capped per episode. A strong moment the tagger doesn't mark is never considered (false negative). |
