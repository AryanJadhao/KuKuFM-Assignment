import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = sys.argv[1]
INK = colors.HexColor("#1E1B18")
RUST = colors.HexColor("#B4441E")
MUTED = colors.HexColor("#6B655C")
RULE = colors.HexColor("#DDD6CA")
PAPER2 = colors.HexColor("#F4F1EB")

base = ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, leading=11.2, textColor=INK, alignment=TA_LEFT)
small = ParagraphStyle("s", parent=base, fontSize=7.8, leading=9.8)
bullet = ParagraphStyle("bl", parent=base, leftIndent=9, bulletIndent=0, spaceAfter=0.6)
h1 = ParagraphStyle("h1", parent=base, fontName="Times-Bold", fontSize=17, leading=20, textColor=INK)
sub = ParagraphStyle("sub", parent=base, fontSize=8.4, textColor=MUTED)
h2 = ParagraphStyle("h2", parent=base, fontName="Times-Bold", fontSize=11.2, leading=13, textColor=RUST,
                    spaceBefore=6.5, spaceAfter=2.5)
cell = ParagraphStyle("c", parent=base, fontSize=7.8, leading=9.6)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")


def P(t, st=base):
    return Paragraph(t, st)


def B(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, head=True):
    data = [[P(c, cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), PAPER2))
    t.setStyle(TableStyle(style))
    return t


W = A4[0] - 3.2 * cm
s = []
s.append(P("AnchorCut: finding promos in microdrama screenplays", h1))
s.append(P("Kuku AI Generalist Assignment · Problem 1 (Find the Promos) · "
           "Aryan Jadhao · Code: github.com/AryanJadhao/KuKuFM-Assignment · Raw outputs: AnchorCut_raw_outputs.zip", sub))
s.append(Spacer(1, 3))

s.append(P("1 · Problem and why I picked it", h2))
s.append(P("Today an editor watches every episode to find promo moments, and Kuku ships a new show every week. "
           "Problem 1 is that editor's job: a repeatable task with a checkable output (episode, start and end "
           "timestamps), so I can measure where the tool is right or wrong instead of judging a chat transcript. It also "
           "splits cleanly between what an LLM is good at (reading drama) and what code must guarantee (20-60 s, hook "
           "in the first 3 s, never cutting a line). The goal is not to replace the editor but to turn 50 episodes of "
           "watching into ~20 suggested clips to approve."))

s.append(P("2 · Assumptions (where the brief was silent)", h2))
s += B([
    "Times are MM:SS.cc from each episode's start. A <b>beat</b> is one tagged block (CAMERA, ACTION, SOUND FX or a "
    "speaker) with its own Start/End. CAMERA beats overlap dialogue, which is normal.",
    "Promo length = sum of (segment end - start), which must be 20.0-60.0 s. Hard cuts between segments take no time.",
    "<b>Hook</b> = the hook beat begins within 3.0 s of the clip start. <b>Never cut a spoken line</b>: a segment cannot "
    "start or end mid-dialogue, and the end also waits for an action in progress to finish.",
    "A multi-moment promo is at most a 3-8 s <b>flash-forward</b> from later in the story (same or next episode) "
    "followed by one continuous scene that ends on a cliffhanger. Wider montages are out of scope.",
    "Only the screenplay is available (no video), so 'does it stop a thumb' is judged from text and stated as an "
    "opinion, not a fact. Malformed timestamps are logged, never silently dropped.",
])

s.append(P("3 · How it works: Anchor → Expand → Verify", h2))
s.append(P("The key decision: <b>the LLM never writes a timestamp or picks a boundary.</b> It only marks single "
           "beats; code does all the time arithmetic, so every cut point is exact and checkable."))
s.append(Spacer(1, 2.5))
s.append(table([
    ["Step", "Who", "What it does"],
    ["1 Parse", "Code", "Regex parser turns 50 screenplays into 3,727 timestamped beats (beat_id, type, speaker, "
                        "text). Every unparseable line becomes a logged issue (0 on this pack)."],
    ["2 Anchors", "LLM, 1 call per episode", "Tags single beats as HOOK_ANCHOR (hook_intensity 1-10) or "
                                             "CLIFFHANGER_ANCHOR (tension 1-10) and quotes the beat. Code checks every "
                                             "quote is really in that beat's text and drops invented ones."],
    ["3 Expand", "Code", "<b>Mode A</b>: hook → later cliffhanger in one stretch, kept if 20-60 s. <b>Mode B</b>: a "
                         "3-8 s flash-forward hook followed by a scene ending on a cliffhanger. Snaps cuts so no line is "
                         "split, removes overlapping windows, keeps ≤ 3 per episode for variety."],
    ["4 Critic", "LLM, 5 clips per call", "A 'cold viewer' who has never seen the show reads each clip exactly as it "
                                          "plays (t = seconds) and scores first-3-s hook, cliffhanger and clarity (1-10), "
                                          "lists confusions, may suggest trims. It never sees why the clip was built."],
    ["5 Verify + tier", "Code", "Re-checks every boundary against the screenplay. Readable rules, not a weighted sum: "
                                "<b>Tier 1</b> post first (hook ≥ 8, clarity ≥ 8, cliffhanger ≥ 7, no prior context "
                                "needed), <b>Tier 2</b> second wave, <b>Tier 3</b> held for a human (low clarity, resolved "
                                "ending, weak hook/cliff). Same-episode posts are spaced apart."],
], [2.1 * cm, 2.8 * cm, W - 4.9 * cm]))
s.append(Spacer(1, 2.5))
s.append(P("Built as a Python package with a Streamlit UI (posting schedule by tier, a timeline inspector showing "
           "the hook zone and cut points, an episode anchor map, live progress, one-click export of the unedited raw "
           "outputs) and a CLI. The model is swapped in .env (Gemini or any OpenAI-compatible endpoint, with a fallback "
           "list); every call is saved unedited; runs resume after quota limits without redoing finished work.", small))

s.append(P("4 · Ranking: what to post first, and why", h2))
s.append(P("Rank = Tier 1 first, ordered by hook, then clarity, then cliffhanger, because cold traffic decides in the first "
           "3 seconds and must follow the story with no backstory. Tier 2 follows, ordered by cliffhanger strength, for "
           "viewers who already know the characters. The first five posts (critic scores hook / cliffhanger / clarity):"))
s.append(Spacer(1, 2.5))
s.append(table([
    ["#", "Segments (in play order)", "Opens on (hook)", "Cuts on (cliffhanger)", "H/C/Cl"],
    ["1", "Ep3 01:09.90-01:16.80 + Ep2 01:03.50-01:36.60 (40.0 s)", "\"Take your hands off her.\"",
     "\"You have shown them exactly what your word is worth.\"", "10/9/8"],
    ["2", "Ep5 00:54.50-01:40.80 (46.3 s)", "\"Barbara, don't!\"", "\"Choose your next words carefully.\"", "9/9/8"],
    ["3", "Ep22 02:03.50-02:07.00 + Ep21 01:04.50-01:41.20 (40.2 s)", "\"This isn't ordinary fever.\"",
     "\"She is dying, Lady Juliana.\"", "9/9/8"],
    ["4", "Ep16 00:14.40-00:52.20 (37.8 s)", "A thug's club caught mid-swing", "\"Gerald Harrington paid us!\"", "9/9/8"],
    ["5", "Ep43 01:09.00-01:52.50 (43.5 s)", "\"You sold my men to the enemy.\"", "\"Going somewhere, Lord Sterling?\"", "9/8/8"],
], [0.5 * cm, 5.6 * cm, 4.2 * cm, 5.4 * cm, W - 15.7 * cm]))
s.append(Spacer(1, 2))
s.append(P("Why #1 leads: a stranger storming in to defend a humiliated bride is an instant hook, and cutting on her "
           "defiant line leaves the question 'who is he, and what happens now?' open. Two of the top three are "
           "flash-forward stitches, a cut an editor would otherwise build by hand. Full ranked list with every hook, "
           "cliffhanger and reason: posting_schedule.csv in the raw outputs.", small))

s.append(P("5 · Scaling to every show Kuku makes", h2))
s += B([
    "<b>Cost is linear and small.</b> The full 50-episode pack took 80 model calls (~1.05 M prompt characters, roughly "
    "260 k input tokens) and ~29 minutes end to end, on the free Gemini tier. Cost per show = 1 call per episode + 1 per "
    "5 candidates.",
    "<b>Episodes are independent</b>, so tagging and critic batches already run in parallel (4 at a time here); a weekly "
    "show is one batch job, and many shows are a queue. No step needs the whole series in one prompt.",
    "<b>The model is a setting</b>: a stronger model Kuku provides plugs in through .env, and the code-side guarantees "
    "(durations, hook timing, no mid-line cuts, grounded quotes) hold whatever model is used.",
    "<b>The human step scales too</b>: an editor reviews ~20 ranked suggestions per show instead of watching 50 episodes, "
    "and their accept/reject decisions become the data to recalibrate the tier thresholds per genre.",
])

s.append(P("6 · How I know it's good, and where it fails", h2))
s.append(P("<b>Results on the full pack:</b> 226 hook and 161 cliffhanger anchors (1 dropped as ungrounded); 624 "
           "windows built; 112 sent to the critic, covering all 50 episodes; <b>11 Tier 1, 62 Tier 2, 39 Tier 3, 0 "
           "rejected</b>. Average critic scores: hook 8.1, cliffhanger 7.7, clarity 7.0."))
s.append(Spacer(1, 2))
s.append(table([
    ["Guaranteed by code (verified)", "Model opinion (not verified)"],
    ["All 112 clips are 24-60 s; every hook starts ≤ 1.7 s in; no clip starts or ends mid-line; every quoted line "
     "exists in the screenplay; timestamps trace back to the source (spot-checked against the raw script).",
     "Whether a hook really stops a scroll, whether a stranger follows the story, and the tier itself. These come from "
     "an LLM judging LLM picks, with no ground truth."],
], [W / 2, W / 2]))
s.append(Spacer(1, 2.5))
s.append(P("<b>Where it picks badly:</b>"))
s += B([
    "<b>Context:</b> the critic flagged relationship / earlier-episode confusion 89 times. Mid-series microdrama "
    "assumes you know who people are, so 27 clips are held in Tier 3 for that confusion and 8 for low clarity overall.",
    "<b>Endings:</b> 14 clips cut after the tension is already resolved. The anchor tagger rates a peak line, but the "
    "next beat sometimes answers it.",
    "<b>Trims unused:</b> the critic was allowed to move boundaries but suggested 0 trims, so all boundaries come from code.",
    "<b>Calibration:</b> the Tier 3 threshold was tuned on this same run (no held-out set), and Gemini 3.5 Flash hit "
    "its daily quota mid-run, so 34 of 50 episodes and all critic calls used Flash-Lite. Scores are less comparable "
    "across that switch.",
    "<b>Not yet validated by a human.</b> The next step is an editor accepting or rejecting Tier 1 and a sample of Tier 3; "
    "that hit rate, not the critic score, is the real quality measure. A screenplay is not footage: acting, music and "
    "pacing can make or break a clip.",
])

s.append(P("7 · AI tools used, and where they got it wrong", h2))
s.append(P("<b>Claude (Claude Code)</b> helped me design and write the code, tests and UI; <b>Gemini 3.5 Flash / "
           "Flash-Lite</b> is the model inside the tool (anchor tagging and the cold-viewer critic)."))
s.append(Spacer(1, 1.5))
s.append(P("<b>Where it went wrong:</b> the first tier rule, written with the AI assistant, sent <i>any</i> clip with a "
           "relationship or earlier-event note to human review. It looked sensible on paper, but on the real run 81 % of "
           "reviewed clips (69 of 85) landed in Tier 3, because the critic attaches such a note to nearly every "
           "mid-series clip even when it rates clarity 7-8/10. A review queue holding 81 % of clips is useless. I "
           "caught it by looking at the reason counts, changed the rule to flag ambiguity only at clarity ≤ 6, and "
           "re-tiered from the saved outputs with no new model calls (on those 85 clips the Tier 1/2/3 split went from 2/14/69 to 9/48/28; "
           "11/62/39 on all 112). Smaller misses: a caching bug made the UI show every promo as unreviewed, and a temporary server "
           "overload (HTTP 503) was first treated like an exhausted quota and paused the whole run."))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm,
                        bottomMargin=1.2 * cm, title="AnchorCut - Solution doc", author="")
doc.build(s)
print("written", OUT)
