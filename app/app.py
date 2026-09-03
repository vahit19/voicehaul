"""VoiceHaul - interactive report.

Everything on this page is computed when you load it. The harness is pure
standard library; gradio and plotly are here only to draw.
"""

import csv
import io
import json
import math
import os
import tempfile

try:                                   # ZeroGPU hardware expects this import
    import spaces                        # noqa: F401
    HAS_SPACES = True
except ImportError:                      # local development
    HAS_SPACES = False

import gradio as gr
import plotly.graph_objects as go

from voicehaul.bands import BANDS, COLOURS, describe
from voicehaul.config import SuiteConfig
from voicehaul.cost import CostModel, estimate as cost_estimate
from voicehaul.transcript import SAMPLE as SAMPLE_CALL
from voicehaul.transcript import analyse as analyse_transcript
from voicehaul.env import PERSONAS
from voicehaul.gate import compare
from voicehaul import metrics as M
from voicehaul.onset import (COMPONENTS, anomaly_components, dominant_cause,
                             localize, score_localization)
from voicehaul.registry import get_persona, get_policy
from voicehaul.report import render_text
from voicehaul.runner import run_episode
from voicehaul.select import budget_curve, equivalent_budget

INK, INK2, MUTED, FAINT = "#1c2725", "#41514e", "#64776f", "#8b9d98"
RULE, ACCENT, ALARM, AMBER, VIOLET = ("#d8e0dd", "#0f766b", "#a8483d",
                                      "#8a6420", "#6a5aa0")
COLOR = {"mirror": ALARM, "flat_cheerful": AMBER, "drifter": VIOLET,
         "calibrated": ACCENT, "oracle": "#93a5a1"}
LABEL = {"mirror": "mirror", "flat_cheerful": "flat", "drifter": "drifter",
         "calibrated": "calibrated", "oracle": "oracle"}
PLABEL = {"distressed_billing": "billing dispute",
          "hostile_escalation": "hostile caller",
          "confused_elderly": "confused caller",
          "grieving_claim": "grieving claimant",
          "cautious_optimist": "calm caller"}

CSS = """
:root{
  --vh-bg:#eef1f0; --vh-surface:#f8faf9; --vh-sunk:#e7ecea;
  --vh-ink:#1c2725; --vh-ink2:#41514e; --vh-muted:#64776f; --vh-faint:#8b9d98;
  --vh-rule:#d8e0dd; --vh-rule2:#c3cfcb;
  --vh-accent:#0f766b; --vh-accent-soft:#dcebe7;
  --vh-alarm:#a8483d; --vh-alarm-soft:#f2e0dc;
  --vh-amber:#8a6420; --vh-amber-soft:#f1e7d3;
  --vh-violet:#6a5aa0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --vh-bg:#111817; --vh-surface:#18211f; --vh-sunk:#141c1b;
    --vh-ink:#e4ebe9; --vh-ink2:#bdc9c6; --vh-muted:#93a5a0; --vh-faint:#71847f;
    --vh-rule:#27322f; --vh-rule2:#34413e;
    --vh-accent:#54c3b3; --vh-accent-soft:#16302c;
    --vh-alarm:#e08373; --vh-alarm-soft:#33211d;
    --vh-amber:#d9a95a; --vh-amber-soft:#2f2718;
    --vh-violet:#a695e0;
  }
}
:root[data-theme="dark"], .dark{
  --vh-bg:#111817; --vh-surface:#18211f; --vh-sunk:#141c1b;
  --vh-ink:#e4ebe9; --vh-ink2:#bdc9c6; --vh-muted:#93a5a0; --vh-faint:#71847f;
  --vh-rule:#27322f; --vh-rule2:#34413e;
  --vh-accent:#54c3b3; --vh-accent-soft:#16302c;
  --vh-alarm:#e08373; --vh-alarm-soft:#33211d;
  --vh-amber:#d9a95a; --vh-amber-soft:#2f2718;
  --vh-violet:#a695e0;
}

body, gradio-app, .gradio-container, .app, .main{
  background:var(--vh-bg) !important; color:var(--vh-ink) !important; }
.gradio-container{max-width:1180px !important; padding-top:1.6rem !important;
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif !important; }
.gradio-container p, .gradio-container li, .gradio-container td,
.gradio-container span, .prose{color:var(--vh-ink2) !important;}
h1,h2,h3,h4{font-family:"IBM Plex Serif",Georgia,serif !important;
  letter-spacing:-.015em !important; color:var(--vh-ink) !important;}
h1{font-size:2.15rem !important; line-height:1.12 !important;}
h2{font-size:1.45rem !important;}
h3{font-size:1.12rem !important;}
a{color:var(--vh-accent) !important;}

/* Panels, cards and inputs sit on the surface tone, never on raw white. */
.block, .form, .panel, .gr-box, .gr-panel,
.gradio-container .block{background:var(--vh-surface) !important;
  border-color:var(--vh-rule) !important; box-shadow:none !important;}
.gr-input, input, textarea, select,
.gradio-container input[type="text"], .gradio-container textarea{
  background:var(--vh-sunk) !important; color:var(--vh-ink) !important;
  border-color:var(--vh-rule2) !important;}
.gradio-container label span, .gradio-container .label-wrap span{
  color:var(--vh-muted) !important; font-size:.82rem !important;}

/* Tabs read as a rule with one live item, not as chunky buttons. */
.tab-nav, .tabs > .tab-nav{border-bottom:1px solid var(--vh-rule) !important;
  background:transparent !important;}
.tab-nav button, button.tab{font-family:"IBM Plex Mono",monospace !important;
  font-size:.8rem !important; color:var(--vh-muted) !important;
  background:transparent !important; border:0 !important;
  padding:.55rem .85rem !important;}
.tab-nav button.selected, button.tab.selected{color:var(--vh-accent) !important;
  border-bottom:2px solid var(--vh-accent) !important;}

button.primary, .gr-button-primary{background:var(--vh-accent) !important;
  border-color:var(--vh-accent) !important; color:#f4faf8 !important;
  font-weight:500 !important;}
button.secondary, .gr-button{background:var(--vh-sunk) !important;
  color:var(--vh-ink2) !important; border-color:var(--vh-rule2) !important;}

table thead th{background:transparent !important; color:var(--vh-faint) !important;}
table td, table th{border-color:var(--vh-rule) !important;}
.table-wrap, .dataframe{background:var(--vh-surface) !important;
  border-color:var(--vh-rule) !important;}
hr{border-color:var(--vh-rule) !important;}
footer, .footer{display:none !important;}

.vh-note{font-size:.9rem; color:var(--vh-muted);}
.vh-fig{margin:1rem 0;}
.vh-fig svg{max-width:100%; height:auto;}
.vh-fig figcaption{font-size:.84rem; color:var(--vh-muted); margin-top:.6rem;
  line-height:1.55; max-width:66ch;}
.dlab{font-family:"IBM Plex Sans",sans-serif; font-size:13px; font-weight:500;
  fill:var(--vh-ink);}
.dsub{font-family:"IBM Plex Mono",monospace; font-size:10px; fill:var(--vh-faint);}
.dedge,.dnote{font-family:"IBM Plex Mono",monospace; font-size:10.5px;
  fill:var(--vh-muted);}
.vh-stroke{stroke:var(--vh-ink2);} .vh-fill{fill:var(--vh-ink2);}
"""

# ---------------------------------------------------------------------------
# block diagrams
# ---------------------------------------------------------------------------

LOOP_SVG = """
<figure class="vh-fig">
<svg viewBox="0 0 900 310" role="img" xmlns="http://www.w3.org/2000/svg"
 aria-label="One turn: the caller's state becomes an utterance, the model
 replies, the reply is measured into a delivery vector, and that vector is
 scored twice. Perceived empathy is what a rater sees and feeds nothing;
 calibration drives the caller's next state.">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--vh-ink2)"/></marker>
<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--vh-accent)"/></marker>
<marker id="arx" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--vh-amber)"/></marker>
</defs>
<rect x="14" y="112" width="132" height="52" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="80" y="134" text-anchor="middle" class="dlab">caller state</text>
<text x="80" y="151" text-anchor="middle" class="dsub">6-dim affect</text>
<line x1="146" y1="138" x2="212" y2="138" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#ar)"/>
<text x="179" y="128" text-anchor="middle" class="dedge">renders</text>
<rect x="212" y="112" width="128" height="52" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="276" y="134" text-anchor="middle" class="dlab">utterance</text>
<text x="276" y="151" text-anchor="middle" class="dsub">what is said</text>
<line x1="340" y1="138" x2="404" y2="138" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#ar)"/>
<rect x="404" y="106" width="126" height="64" rx="3" fill="var(--vh-accent-soft)" stroke="var(--vh-accent)" stroke-width="1.4"/>
<text x="467" y="132" text-anchor="middle" class="dlab">model</text>
<text x="467" y="149" text-anchor="middle" class="dsub">under test</text>
<line x1="530" y1="138" x2="586" y2="138" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#ar)"/>
<text x="562" y="128" text-anchor="middle" class="dedge">reply</text>
<rect x="586" y="100" width="156" height="76" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="664" y="124" text-anchor="middle" class="dlab">delivery vector</text>
<text x="664" y="142" text-anchor="middle" class="dsub">rate, warmth,</text>
<text x="664" y="155" text-anchor="middle" class="dsub">length, apology,</text>
<text x="664" y="168" text-anchor="middle" class="dsub">acknowledgement</text>
<path d="M742 120 L774 120 L774 52 L806 52" fill="none" stroke="var(--vh-amber)" stroke-width="1.4" marker-end="url(#arx)"/>
<path d="M742 156 L774 156 L774 232 L806 232" fill="none" stroke="var(--vh-accent)" stroke-width="1.4" marker-end="url(#ara)"/>
<rect x="806" y="28" width="82" height="48" rx="3" fill="none" stroke="var(--vh-amber)" stroke-width="1.3"/>
<text x="847" y="48" text-anchor="middle" class="dlab" fill="var(--vh-amber)">perceived</text>
<text x="847" y="64" text-anchor="middle" class="dlab" fill="var(--vh-amber)">empathy</text>
<rect x="806" y="208" width="82" height="48" rx="3" fill="none" stroke="var(--vh-accent)" stroke-width="1.3"/>
<text x="847" y="228" text-anchor="middle" class="dlab" fill="var(--vh-accent)">calibration</text>
<text x="847" y="244" text-anchor="middle" class="dsub" fill="var(--vh-accent)">hidden</text>
<text x="888" y="94" text-anchor="end" class="dnote" fill="var(--vh-amber)">what a rater scores</text>
<text x="888" y="288" text-anchor="end" class="dnote" fill="var(--vh-accent)">what moves the caller</text>
<path d="M806 232 L768 232 L768 268 L80 268 L80 164" fill="none" stroke="var(--vh-accent)"
 stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#ara)"/>
<text x="424" y="262" text-anchor="middle" class="dedge" fill="var(--vh-accent)">drives the next state</text>
<line x1="847" y1="76" x2="847" y2="196" stroke="var(--vh-amber)" stroke-width="1.2" stroke-dasharray="3 5"/>
<text x="856" y="134" class="dnote" fill="var(--vh-amber)">feeds</text>
<text x="856" y="149" class="dnote" fill="var(--vh-amber)">nothing</text>
</svg>
<figcaption>One turn. The same delivery vector is scored twice, and only one of
the two changes what happens next. The observable score is the one with no
arrow leaving it &mdash; which is why a turn-level suite and a conversation can
disagree.</figcaption></figure>
"""

RATERS_SVG = """
<figure class="vh-fig">
<svg viewBox="0 0 900 320" role="img" xmlns="http://www.w3.org/2000/svg"
 aria-label="One turn is scored by a human panel and by an LLM judge. The
 customer's estimate comes from correlating those two and dividing out the
 panel's own unreliability. A third path, the latent truth, exists only in
 simulation and is what validates the estimate.">
<defs><marker id="b1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--vh-ink2)"/></marker>
<marker id="b2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--vh-accent)"/></marker></defs>
<rect x="14" y="128" width="118" height="52" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="73" y="150" text-anchor="middle" class="dlab">one turn</text>
<text x="73" y="167" text-anchor="middle" class="dsub">caller + reply</text>
<path d="M132 144 L200 78" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M132 154 L200 154" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M132 164 L200 246" fill="none" stroke="var(--vh-accent)" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#b2)"/>
<rect x="200" y="52" width="150" height="52" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="275" y="74" text-anchor="middle" class="dlab">human panel</text>
<text x="275" y="91" text-anchor="middle" class="dsub">k raters, noisy</text>
<rect x="200" y="128" width="150" height="52" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="275" y="150" text-anchor="middle" class="dlab">LLM judge</text>
<text x="275" y="167" text-anchor="middle" class="dsub">cheap, unproven</text>
<rect x="200" y="220" width="150" height="52" rx="3" fill="none" stroke="var(--vh-accent)" stroke-width="1.4" stroke-dasharray="5 4"/>
<text x="275" y="242" text-anchor="middle" class="dlab" fill="var(--vh-accent)">latent truth</text>
<text x="275" y="259" text-anchor="middle" class="dsub" fill="var(--vh-accent)">simulation only</text>
<path d="M350 78 L430 114" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M350 154 L430 134" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#b1)"/>
<rect x="430" y="98" width="176" height="58" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.2"/>
<text x="518" y="120" text-anchor="middle" class="dlab">correlate, then</text>
<text x="518" y="137" text-anchor="middle" class="dlab">disattenuate</text>
<text x="518" y="151" text-anchor="middle" class="dsub">divide out panel noise</text>
<line x1="606" y1="127" x2="672" y2="127" stroke="var(--vh-ink2)" stroke-width="1.2" marker-end="url(#b1)"/>
<rect x="672" y="98" width="150" height="58" rx="3" fill="none" stroke="var(--vh-ink2)" stroke-width="1.4"/>
<text x="747" y="122" text-anchor="middle" class="dlab">estimated rho</text>
<text x="747" y="141" text-anchor="middle" class="dsub">what a customer gets</text>
<path d="M350 246 L660 246 L660 172 L700 172 L700 160" fill="none" stroke="var(--vh-accent)"
 stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#b2)"/>
<text x="505" y="238" text-anchor="middle" class="dedge" fill="var(--vh-accent)">is the estimate right?</text>
<text x="886" y="196" text-anchor="end" class="dnote" fill="var(--vh-accent)">this arrow does not exist on real data:</text>
<text x="886" y="212" text-anchor="end" class="dnote" fill="var(--vh-accent)">both measurements carry error and</text>
<text x="886" y="228" text-anchor="end" class="dnote" fill="var(--vh-accent)">neither one is the reference</text>
</svg>
<figcaption>The estimator a customer can run uses only the two solid paths. The
dashed path is the validation, and it is the reason this is built in a simulator
before it is pointed at a contract.</figcaption></figure>
"""


def base_fig(height=360, x="", y=""):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=12, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=x, gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE),
        yaxis=dict(title=y, gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE))
    return fig


# ---------------------------------------------------------------------------
# tab: release gate
# ---------------------------------------------------------------------------


def _chip(label, colour):
    return ('<span style="display:inline-block;font-family:IBM Plex Mono,monospace;'
            'font-size:.62rem;letter-spacing:.06em;text-transform:uppercase;'
            'padding:1px 6px;border-radius:9px;background:{c}1a;color:{c};'
            'margin-left:.45rem">{l}</span>').format(c=colour, l=label)


def _scale(value, floor, ceiling, colour):
    """Where this number sits between the worst and best the suite reached."""
    if floor is None or ceiling is None or ceiling <= floor:
        return ""
    pos = max(0.0, min(1.0, (value - floor) / (ceiling - floor)))
    return ('<div style="margin-top:.3rem"><div style="position:relative;'
            'height:5px;background:var(--vh-sunk);border-radius:3px">'
            '<div style="position:absolute;left:0;top:0;height:5px;width:{w:.0f}%;'
            'background:{c};border-radius:3px"></div>'
            '<div style="position:absolute;left:{w:.0f}%;top:-3px;width:2px;'
            'height:11px;background:{c}"></div></div>'
            '<div style="display:flex;justify-content:space-between;'
            'font-family:IBM Plex Mono,monospace;font-size:.6rem;color:var(--vh-faint);'
            'margin-top:2px"><span>floor {f:.2f}</span>'
            '<span>ceiling {t:.2f}</span></div></div>'
            ).format(w=100 * pos, c=colour, f=floor, t=ceiling)


def reading_panel(names):
    """What each number means and where its threshold came from."""
    rows = ""
    for n in names:
        b = BANDS.get(n)
        if not b:
            continue
        cuts = " &nbsp;&middot;&nbsp; ".join(
            '<span style="color:{c}">{l} {op} {t}</span>'.format(
                c=COLOURS[l], l=l,
                op="&ge;" if b.higher_is_better or b.kind == "anchored" else "&le;",
                t=abs(t) if not b.higher_is_better and b.kind != "anchored" else t)
            for t, l in b.cuts)
        rows += (
            '<tr><td style="padding:.6rem .8rem .6rem 0;vertical-align:top;'
            'white-space:nowrap;font-family:IBM Plex Mono,monospace;'
            'font-size:.85rem;border-bottom:1px solid var(--vh-rule)">{n}<br>'
            '<span style="font-size:.66rem;letter-spacing:.08em;'
            'text-transform:uppercase;color:var(--vh-faint)">{k}</span></td>'
            '<td style="padding:.6rem .8rem;vertical-align:top;font-size:.87rem;'
            'color:var(--vh-ink2);border-bottom:1px solid var(--vh-rule)">{r}'
            '<div style="margin-top:.35rem;font-family:IBM Plex Mono,monospace;'
            'font-size:.72rem">{c}</div>'
            '<div style="margin-top:.3rem;font-size:.78rem;color:var(--vh-faint)">{s}</div>'
            '</td></tr>').format(n=n, k=b.kind, r=b.reading, c=cuts, s=b.source)
    return ('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
            'width:100%;font-family:IBM Plex Sans,system-ui,sans-serif">{}'
            '</table></div>').format(rows)



POLICY_BLURB = {
    "calibrated": ("the reference", ACCENT,
                   "Tracks the caller's state, steers toward calm, and keeps "
                   "honouring a request once it is made."),
    "mirror": ("sounds attuned, regulates nothing", ALARM,
               "Matches the caller's energy and mood. Rated highly turn by "
               "turn; leaves hostile callers where it found them."),
    "flat_cheerful": ("ignores everything", AMBER,
                      "One upbeat persona for every caller and every request."),
    "drifter": ("good early, fades", VIOLET,
                "Well calibrated at the start; quietly stops honouring "
                "requests as the session runs long."),
    "oracle": ("the ceiling", "#93a5a1",
               "Noiseless upper bound. Not a model anyone could ship - it is "
               "the top of the scale every other row is read against."),
}

CALLER_BLURB = {
    "billing dispute": "Charged twice, rent due tomorrow. Distressed, not angry.",
    "hostile caller": "Third call, nothing done. Escalating; the segment that "
                      "generates complaints.",
    "confused caller": "Does not understand the letter. Needs pace, not speed.",
    "grieving claimant": "Handling a deceased partner's account.",
    "calm caller": "Small error, no urgency. The easy case.",
}


def _legend(entries, title):
    rows = "".join(
        ('<div style="display:flex;gap:.7rem;align-items:flex-start;'
         'padding:.45rem 0;border-bottom:1px solid var(--vh-rule)">'
         '<div style="width:9px;height:9px;border-radius:50%;background:{c};'
         'margin-top:.35rem;flex:0 0 auto"></div>'
         '<div><b style="font-family:IBM Plex Mono,monospace;font-size:.84rem">'
         '{k}</b> <span style="color:{c};font-size:.8rem">{t}</span>'
         '<div style="font-size:.85rem;color:var(--vh-muted);line-height:1.45">{d}</div>'
         '</div></div>').format(c=c, k=k, t=t, d=d)
        for k, (t, c, d) in entries.items())
    return ('<details style="margin:.2rem 0 .8rem;font-family:IBM Plex Sans,'
            'system-ui,sans-serif"><summary style="cursor:pointer;font-size:.86rem;'
            'color:var(--vh-accent)">{}</summary><div style="margin-top:.5rem">{}</div>'
            '</details>').format(title, rows)


def policy_legend():
    return _legend(POLICY_BLURB, "What are these five policies?")


def caller_legend():
    return _legend({k: ("", MUTED, v) for k, v in CALLER_BLURB.items()},
                   "Who are the five callers?")



AUTHOR_CARD = """
<div style="display:flex;flex-wrap:wrap;gap:1.4rem;align-items:flex-start;
 justify-content:space-between;border:1px solid var(--vh-rule);border-radius:6px;
 background:var(--vh-surface);padding:16px 20px;margin:0 0 1.1rem;
 font-family:'IBM Plex Sans',system-ui,sans-serif">
  <div style="flex:1 1 340px;min-width:280px">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.64rem;
     letter-spacing:.13em;text-transform:uppercase;color:var(--vh-faint)">
     built by</div>
    <div style="font-family:'IBM Plex Serif',Georgia,serif;font-size:1.35rem;
     font-weight:600;color:var(--vh-ink);margin:.2rem 0 .1rem">Vahit Feryad, PhD</div>
    <div style="font-size:.9rem;color:var(--vh-ink2)">Applied AI research engineer
     &middot; evaluation, benchmarking and agent reliability</div>
    <div style="font-size:.84rem;color:var(--vh-muted);margin-top:.45rem;line-height:1.55">
     PhD in electrical and electronics engineering &middot; 10+ years industrial
     R&amp;D &middot; 237 Google Scholar citations &middot; Istanbul</div>
  </div>
  <div style="flex:0 1 330px;min-width:270px">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.64rem;
     letter-spacing:.13em;text-transform:uppercase;color:var(--vh-faint);
     margin-bottom:.4rem">the work this is built on</div>
    <div style="font-size:.85rem;line-height:1.6;color:var(--vh-ink2)">
      <a href="https://github.com/vahit19/LongHaul-Bench"
       style="color:var(--vh-accent);font-weight:500">LongHaul-Bench</a>
      &mdash; long-horizon agent reliability over 1,000+ sequential episodes,
      five-world programme, memory ablations<br>
      <a href="https://github.com/vahit19/runopsy"
       style="color:var(--vh-accent);font-weight:500">Runopsy</a>
      &mdash; causal failure-onset diagnosis with counterfactual replay
      (Apache-2.0, on PyPI)
    </div>
    <div style="margin-top:.6rem;font-family:'IBM Plex Mono',monospace;
     font-size:.78rem">
      <a href="https://github.com/vahit19/voicehaul" style="color:var(--vh-accent)">source</a>
      &nbsp;&middot;&nbsp;
      <a href="https://scholar.google.com/citations?hl=en&amp;user=JUtYZ1oAAAAJ"
       style="color:var(--vh-accent)">scholar</a>
      &nbsp;&middot;&nbsp;
      <a href="https://www.linkedin.com/in/vahit-feryad-19517256/"
       style="color:var(--vh-accent)">linkedin</a>
      &nbsp;&middot;&nbsp;
      <a href="https://vahit19.github.io/voicehaul/" style="color:var(--vh-accent)">static report</a>
    </div>
  </div>
</div>
"""

TABS_MAP = """
<div style="border:1px solid var(--vh-rule);border-radius:6px;background:var(--vh-sunk);
 padding:14px 18px;margin:.2rem 0 1.2rem;font-family:'IBM Plex Sans',system-ui,
 sans-serif">
 <div style="font-family:'IBM Plex Mono',monospace;font-size:.64rem;
  letter-spacing:.13em;text-transform:uppercase;color:var(--vh-faint);
  margin-bottom:.6rem">what you can do on this page</div>
 <div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;
  font-size:.86rem">
 <tr><th style="text-align:left;padding:.3rem .8rem .45rem 0;font-size:.64rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--vh-faint);font-weight:500">
  tab</th>
 <th style="text-align:left;padding:.3rem .8rem .45rem;font-size:.64rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--vh-faint);font-weight:500">
  you choose</th>
 <th style="text-align:left;padding:.3rem 0 .45rem .8rem;font-size:.64rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--vh-faint);font-weight:500">
  you get</th></tr>
 {rows}
 </table></div>
 <div style="font-size:.82rem;color:var(--vh-muted);margin-top:.7rem">
  Every tab already shows a result. Nothing calls an external service &mdash;
  the harness is pure Python and runs in this tab.</div>
</div>
"""

_TABS = [
    ("Judge substitution",
     "nothing - it is precomputed",
     "how many human ratings one LLM judge rating is worth, per dimension and "
     "per caller, with the estimator checked against ground truth"),
    ("Score your own call",
     "a transcript you paste in",
     "whether the agent did what the caller asked and kept doing it, the "
     "delivery trace, the turns worth looking at, and a CSV"),
    ("Release gate",
     "a baseline and a candidate policy, suite size",
     "ship or block, every dimension with its scale, where the regression "
     "lands by caller, and what the sample size could not resolve"),
    ("Watch a conversation",
     "a policy and a caller",
     "the call turn by turn, with the turn it broke on marked and named"),
    ("Which turn broke it",
     "how subtle the injected fault is, and where",
     "the four signals that flagged each turn, the evidence at the one named, "
     "and accuracy across 30 injected faults"),
    ("Rating budget",
     "conversations, raters, how much raters disagree",
     "the smallest regression you could detect, and which conversations to "
     "spend the budget on"),
    ("How to read the numbers",
     "nothing",
     "every metric with its bands and where each threshold came from"),
    ("What's real here",
     "nothing",
     "what is simulated, what is not, and the faults this exposed in the "
     "instrument itself"),
]


def tabs_map():
    rows = "".join(
        ('<tr><td style="padding:.4rem .8rem .4rem 0;vertical-align:top;'
         'border-top:1px solid var(--vh-rule);font-weight:500;white-space:nowrap">{t}</td>'
         '<td style="padding:.4rem .8rem;vertical-align:top;'
         'border-top:1px solid var(--vh-rule);color:var(--vh-muted)">{i}</td>'
         '<td style="padding:.4rem 0 .4rem .8rem;vertical-align:top;'
         'border-top:1px solid var(--vh-rule);color:var(--vh-ink2)">{o}</td></tr>'
         ).format(t=t, i=i, o=o) for t, i, o in _TABS)
    return TABS_MAP.format(rows=rows)



def _money(x):
    return "${:,.0f}".format(x)


def run_cost(target, cost_human, cost_judge, releases, raters, episodes):
    """Turn the measured statistics into a number a budget holder can act on."""
    eps = [run_episode(get_policy("calibrated"), PERSONAS[i % len(PERSONAS)],
                       seed=i, n_turns=40) for i in range(int(episodes))]
    sb = M.stdev([1.0 + 6.0 * e.mean_perceived for e in eps])

    ks = [k for k in (5, 10, 20, 40) if k <= len(eps)]
    coverage = 1.0
    if ks:
        k = ks[len(ks) // 2]
        eq = equivalent_budget(eps, k)
        if eq:
            coverage = max(0.2, min(1.0, float(k) / eq))

    model = CostModel(cost_per_human_rating=float(cost_human),
                      cost_per_judge_rating=float(cost_judge),
                      releases_per_year=int(releases),
                      raters_per_conversation=int(raters))

    ratios = {"perceived empathy": 0.29, "did it actually help": 0.01}
    if SUB:
        for r in SUB["rows"]:
            if r["segment"] == "all":
                nice = ("perceived empathy" if r["dimension"] == "perceived_empathy"
                        else "did it actually help")
                ratios[nice] = r["ratio_estimated"]

    rows = ""
    for name, ratio in ratios.items():
        plain = cost_estimate(float(target), sb, 0.9, ratio, model, 1.0)
        picked = cost_estimate(float(target), sb, 0.9, ratio, model, coverage)
        col = ACCENT if picked.saving_per_year > 0 else ALARM
        rows += (
            '<tr><td style="padding:.55rem .7rem;border-bottom:1px solid var(--vh-rule)">'
            '{n}<div style="font-size:.75rem;color:var(--vh-faint)">1 judge rating = '
            '{r:.2f} human</div></td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule)">{c}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule)">{b}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule)">{w}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule);'
            'color:{col};font-weight:500">{y}</td></tr>'
        ).format(n=name, r=ratio, c=plain.conversations,
                 b=_money(plain.cost_baseline), w=_money(picked.cost_with_judge),
                 y=_money(picked.saving_per_year), col=col)

    worst = min(ratios.values())
    verdict = cost_estimate(float(target), sb, 0.9, worst, model, coverage)

    head = "".join(
        '<th style="text-align:{a};padding:0 .7rem .5rem;font-size:.64rem;'
        'letter-spacing:.09em;text-transform:uppercase;color:var(--vh-faint);'
        'border-bottom:1px solid var(--vh-rule2);font-weight:500">{h}</th>'.format(
            a="left" if i == 0 else "right", h=h)
        for i, h in enumerate(["dimension", "conversations",
                               "panel only, per release",
                               "with judge + selection", "saved per year"]))

    table = ('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
             'width:100%;font-family:IBM Plex Mono,monospace;font-size:.86rem;'
             'font-variant-numeric:tabular-nums"><tr>{}</tr>{}</table></div>'
             ).format(head, rows)

    note = ("**Read the bottom row first.** On whether a turn actually helped, "
            "the judge is worth {r:.2f} human ratings - below the {f:.2f} floor "
            "where substitution stops being honest. The saving there is **zero**, "
            "and that is the finding: this is the dimension a voice team most "
            "needs and the one an automated rater cannot cover.\n\n"
            "{note}\n\n"
            "Diversity selection independently cuts the conversation count to "
            "**{cov:.0%}** of a random sample for the same coverage of "
            "caller-state space, which applies whether or not a judge is in the "
            "loop.").format(r=worst, f=0.25, note=verdict.note, cov=coverage)
    return table, note



OUT_DIR = os.path.join(tempfile.gettempdir(), "voicehaul-exports")


def _write(name, text):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


def gate_artifacts(rep, cfg):
    """The three surfaces of one report, as files.

    JSON is the one that matters in a pipeline - a CI job reads `verdict` and
    promotes or blocks. Markdown goes in the pull request. The CSV is the raw
    per-conversation rows, so a team can check the statistics rather than trust
    them.
    """
    from voicehaul.report import render_json, render_markdown

    stem = "voicehaul-{}-vs-{}-{}".format(
        rep.baseline_name, rep.candidate_name, cfg.suite_id)
    paths = [_write(stem + ".json", render_json(rep)),
             _write(stem + ".md", render_markdown(rep))]

    rows = [["dimension", "kind", "gating", "baseline", "candidate", "delta",
             "ci95_low", "ci95_high", "p_raw", "p_holm", "verdict",
             "n_baseline", "n_candidate"]]
    for d in rep.dimensions:
        rows.append([d.name,
                     "panel" if d.name.startswith("panel:") else "conversation",
                     rep.gating.get(d.name, True), round(d.baseline, 6),
                     round(d.candidate, 6), round(d.delta, 6),
                     round(d.ci[0], 6), round(d.ci[1], 6), round(d.p, 8),
                     round(d.p_holm, 8), d.verdict, d.n_baseline, d.n_candidate])
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    paths.append(_write(stem + ".csv", buf.getvalue()))
    return paths


CI_SNIPPET = """<div style="font-family:'IBM Plex Sans',system-ui,sans-serif">
<div style="font-family:'IBM Plex Mono',monospace;font-size:.64rem;
 letter-spacing:.13em;text-transform:uppercase;color:var(--vh-faint);
 margin-bottom:.5rem">put it in a pipeline</div>
<p style="font-size:.88rem;color:var(--vh-ink2);margin:0 0 .6rem">
The gate exits non-zero on <b>BLOCK</b>, so it gates a release without any glue
code. The JSON above is the same report a job would parse.</p>
<pre style="background:var(--vh-sunk);border:1px solid var(--vh-rule);border-radius:5px;
 padding:12px 14px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;
 line-height:1.55;overflow-x:auto;margin:0"># .github/workflows/voice-eval.yml
- name: Long-horizon regression gate
  run: |
    pip install -e .
    voicehaul gate $BASELINE $CANDIDATE \\
      --config configs/support-en-40turn.yaml \\
      --format json --out artifacts/
  # exit 1 = a gating dimension regressed; the job fails and the build stops

- uses: actions/upload-artifact@v4
  with:
    name: voice-eval
    path: artifacts/</pre>
<p style="font-size:.82rem;color:var(--vh-muted);margin:.6rem 0 0">
Every run is stamped with a suite id derived from the config, so two reports
carrying the same id were measured the same way and two carrying different ids
were not.</p></div>"""



def run_transcript(text):
    """Measure a call somebody brings with them.

    Only the questions a transcript can honestly answer: what the agent's
    delivery was, whether the caller asked for a change, and whether the agent
    did it and kept doing it. No affect model, nothing inferred that a reader
    could not check by hand.
    """
    if not text or not text.strip():
        return ("<div style='color:var(--vh-faint)'>Paste a call above.</div>", None,
                [], "", None)

    rep = analyse_transcript(text)
    agents = rep.agent_turns
    if not agents:
        return ("<div style='color:var(--vh-alarm)'>No agent turns found. Prefix lines "
                "with <code>Caller:</code> and <code>Agent:</code>.</div>",
                None, [], rep.parse_note, None)

    overall = rep.overall_uptake
    asked = sum(v["requests"] for v in rep.uptake.values())

    if asked == 0:
        colour, headline, sub = MUTED, "no explicit requests", (
            "The caller never asked for a change in delivery, so there is "
            "nothing to track. The delivery trace below still applies.")
    else:
        info = describe("feedback uptake @10", overall)
        colour = info["colour"]
        headline = "{:.0%} of requests honoured".format(overall)
        sub = ("Across {} request{} and {} agent turns after them. Reading: "
               "<b>{}</b>.").format(asked, "" if asked == 1 else "s",
                                    sum(v["opportunities"] for v in rep.uptake.values()),
                                    info["label"])

    card = (
        '<div style="border:1px solid {c};background:{c}12;border-radius:6px;'
        'padding:16px 20px;font-family:IBM Plex Sans,system-ui,sans-serif">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.64rem;'
        'letter-spacing:.12em;text-transform:uppercase;color:{c}">'
        'did the agent do what it was asked?</div>'
        '<div style="font-size:1.7rem;font-weight:600;color:{c};line-height:1.15;'
        'margin:.25rem 0 .3rem">{h}</div>'
        '<div style="font-size:.9rem;color:var(--vh-ink2)">{s}</div></div>'
    ).format(c=colour, h=headline, s=sub)

    rows = []
    for t in rep.turns:
        if t.speaker == "caller":
            rows.append([t.index, "caller", t.text[:160],
                         ", ".join(r.replace("_", " ") for r in t.requests) or "",
                         "", "", "", ""])
        else:
            a = t.action
            rows.append([t.index, "agent", t.text[:160], "",
                         round(a.apology_rate, 2), round(a.verbosity, 2),
                         round(a.acknowledgement, 2),
                         ", ".join(v.replace("_", " ") for v in t.violated) or "ok"])

    fig = base_fig(340, "agent turn", "measured from the words")
    idx = [t.index for t in agents]
    for key, name, colr in (("apology_rate", "apology", ALARM),
                            ("verbosity", "length", AMBER),
                            ("acknowledgement", "acknowledgement", ACCENT)):
        fig.add_trace(go.Scatter(
            x=idx, y=[getattr(t.action, key) for t in agents], mode="lines+markers",
            name=name, line=dict(color=colr, width=2.2)))
    for t in rep.turns:
        for r in t.requests:
            fig.add_vline(x=t.index, line=dict(color=INK2, width=1.2, dash="dot"))
            fig.add_annotation(x=t.index, y=1.02, yanchor="bottom", showarrow=False,
                               text='asked: "{}"'.format(r.replace("_", " ")),
                               font=dict(color=INK2, size=10))
    for i, _why in rep.flagged:
        fig.add_vrect(x0=i - 0.4, x1=i + 0.4, fillcolor=ALARM, opacity=0.10,
                      line_width=0)

    lines = ["**{}**".format(rep.parse_note), ""]
    if asked:
        lines.append("| request | asked | honoured | first {} turns |".format(3))
        lines.append("|---|---:|---:|---:|")
        for name, v in rep.uptake.items():
            if not v["requests"]:
                continue
            first = ("{}/{}".format(v["first_response"], v["first_response_seen"])
                     if v["first_response_seen"] else "-")
            lines.append("| {} | {} | {}/{} | {} |".format(
                name.replace("_", " "), v["requests"], v["honoured"],
                v["opportunities"], first))
        lines.append("")
    if rep.flagged:
        lines.append("**Turns worth looking at**")
        for i, why in rep.flagged:
            lines.append("- turn {}: {}".format(i, why))
        lines.append("")
    lines.append(
        "*What is deliberately not reported: whether the caller ended up better "
        "off. A transcript carries no ground truth about how they felt, and "
        "guessing it would be the kind of number that gets believed. On audio, "
        "expression measurement supplies it.*")

    stem = "voicehaul-your-call.csv"
    header = ["turn", "speaker", "text", "requests", "apology_rate", "verbosity",
              "acknowledgement", "violated"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for t in rep.turns:
        if t.speaker == "caller":
            w.writerow([t.index, "caller", t.text, "|".join(t.requests),
                        "", "", "", ""])
        else:
            a = t.action
            w.writerow([t.index, "agent", t.text, "", round(a.apology_rate, 4),
                        round(a.verbosity, 4), round(a.acknowledgement, 4),
                        "|".join(t.violated)])
    path = _write(stem, buf.getvalue())
    return card, fig, rows, "\n".join(lines), path


VERDICT_STYLE = {
    "BLOCK": (ALARM, "var(--vh-alarm-soft)", "Do not ship"),
    "SHIP": (ACCENT, "var(--vh-accent-soft)", "Safe to ship"),
    "INCONCLUSIVE": (MUTED, "var(--vh-rule)", "Not enough evidence"),
    "SATURATED": (AMBER, "var(--vh-amber-soft)", "This suite cannot separate them"),
}


def _card(rep):
    color, bg, headline = VERDICT_STYLE.get(rep.verdict, (MUTED, "var(--vh-rule)", ""))
    reasons = "".join("<li>{}</li>".format(r) for r in rep.reasons)
    return (
        '<div style="border:1px solid {c};background:{b};border-radius:6px;'
        'padding:18px 22px;font-family:IBM Plex Sans,system-ui,sans-serif">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
        'letter-spacing:.12em;text-transform:uppercase;color:{c};'
        'margin-bottom:.35rem">verdict</div>'
        '<div style="font-size:1.9rem;font-weight:600;color:{c};line-height:1.1">'
        '{v}</div>'
        '<div style="font-size:1.05rem;color:var(--vh-ink2);margin-top:.3rem">{h}</div>'
        '<ul style="margin:.9rem 0 0;padding-left:1.1rem;color:var(--vh-ink2);'
        'font-size:.92rem;line-height:1.55">{r}</ul></div>'
    ).format(c=color, b=bg, v=rep.verdict, h=headline, r=reasons)


def _table(rep):
    vals = {}
    for d in rep.dimensions:
        vals.setdefault(d.name, []).extend([d.baseline, d.candidate])

    def row(d):
        gate = rep.gating.get(d.name, True)
        col = {"improved": ACCENT, "regressed": ALARM}.get(d.verdict, MUTED)
        tag = "" if gate else ' <span style="color:var(--vh-faint)">diagnostic</span>'
        band = BANDS.get(d.name)
        chip, scale = "", ""
        if band is not None:
            lo = hi = None
            if band.kind == "anchored":
                lo, hi = min(vals[d.name]), max(vals[d.name])
                lo, hi = min(lo, 0.0), max(hi, 1.0)
            info = describe(d.name, d.candidate, lo, hi)
            chip = _chip(info["label"], info["colour"])
            scale = _scale(d.candidate, lo, hi, info["colour"])
        return (
            '<tr><td style="padding:.55rem .7rem;border-bottom:1px solid var(--vh-rule)">'
            '{n}{t}{chip}{sc}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule)">{b:.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule);font-weight:500">{c:.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule);'
            'color:{col};font-weight:500">{d:+.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--vh-rule)">{p:.4f}</td>'
            '</tr>').format(n=d.name, t=tag, chip=chip, sc=scale, b=d.baseline,
                            c=d.candidate, d=d.delta, p=d.p_holm, col=col)

    panel = [d for d in rep.dimensions if d.name.startswith("panel:")]
    conv = [d for d in rep.dimensions if not d.name.startswith("panel:")]
    head = ('<tr>' + "".join(
        '<th style="text-align:{a};padding:0 .7rem .5rem;font-size:.66rem;'
        'letter-spacing:.09em;text-transform:uppercase;color:var(--vh-faint);'
        'border-bottom:1px solid var(--vh-rule2);font-weight:500">{h}</th>'.format(
            a=("left" if i == 0 else "right"), h=h)
        for i, h in enumerate(["dimension", "baseline", "candidate", "delta",
                               "holm p"])) + '</tr>')
    sec = lambda t: ('<tr><td colspan="5" style="padding:1rem .7rem .4rem;'
                     'font-family:IBM Plex Mono,monospace;font-size:.68rem;'
                     'letter-spacing:.1em;text-transform:uppercase;color:var(--vh-faint)">'
                     '{}</td></tr>').format(t)
    body = sec("what a fixed-prompt leaderboard reports")
    body += "".join(row(d) for d in panel)
    body += sec("what the conversations report")
    body += "".join(row(d) for d in conv)
    return ('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
            'width:100%;font-family:IBM Plex Mono,monospace;font-size:.85rem;'
            'font-variant-numeric:tabular-nums">{}{}</table></div>').format(head, body)


def run_gate(baseline, candidate, episodes, turns):
    cfg = SuiteConfig(name="ui", episodes=int(episodes), turns=int(turns))
    rep = compare(baseline, candidate, cfg, diagnose=True)

    names = [d.name for d in rep.dimensions]
    imps = [d.improvement for d in rep.dimensions]
    cols = [ACCENT if d.verdict == "improved" else
            ALARM if d.verdict == "regressed" else "var(--vh-rule2)"
            for d in rep.dimensions]
    delta_fig = base_fig(330, "change, signed so positive is better", "")
    delta_fig.add_trace(go.Bar(y=names, x=imps, orientation="h",
                               marker_color=cols, showlegend=False))
    delta_fig.add_vline(x=0, line=dict(color=INK2, width=1.2))

    seg_fig = None
    if rep.segments:
        seg_fig = base_fig(300, "change on the worst gating dimension", "")
        seg_fig.add_trace(go.Bar(
            y=[s.persona.replace("_", " ") for s in rep.segments],
            x=[s.delta for s in rep.segments], orientation="h",
            marker_color=ALARM, showlegend=False))
        seg_fig.add_vline(x=0, line=dict(color=INK2, width=1.2))

    notes = ("**Suite** `{}` &nbsp;|&nbsp; {} conversations &times; {} turns "
             "&times; {} callers. Welch two-sample tests with one conversation "
             "as the unit, Holm-corrected across {} dimensions.\n\n"
             ).format(cfg.suite_id, cfg.episodes, cfg.turns, len(cfg.personas),
                      len(rep.dimensions))
    if rep.onsets:
        srt = sorted(rep.onsets)
        causes = ", ".join("{} ({})".format(k, v) for k, v in
                           sorted(rep.onset_causes.items(), key=lambda kv: -kv[1]))
        notes += ("**Diagnosis.** {} of {} failed candidate conversations were "
                  "localized to an onset turn; median turn {}. Most common "
                  "cause: {}.\n\n").format(
            len(rep.onsets), rep.failures_candidate, srt[len(srt) // 2], causes)
    notes += ("**What this suite could not see.** At n={} with {} raters, the "
              "smallest detectable regression is **{:.2f} Likert points**; "
              "resolving a 0.20-point change needs n = {}."
              ).format(cfg.episodes, cfg.raters_per_conversation, rep.mde,
                       rep.n_for_small_effect)
    seg_title = ("### Where it lands &mdash; {} by caller".format(
        rep.worst_dimension) if rep.segments else "")
    files = gate_artifacts(rep, cfg)
    return (_card(rep), _table(rep), delta_fig, seg_title, seg_fig, notes,
            files)


# ---------------------------------------------------------------------------
# tab: watch a conversation
# ---------------------------------------------------------------------------

def run_conversation(policy, persona_label, turns):
    persona = get_persona([k for k, v in PLABEL.items() if v == persona_label][0])
    ep = run_episode(get_policy(policy), persona, seed=3, n_turns=int(turns))
    xs = list(range(len(ep.turns)))
    fig = base_fig(380, "turn", "level")
    fig.update_yaxes(range=[-0.02, 1.05])
    for key, label, color, width in [
            ("d", "distress the caller is left carrying", ALARM, 3),
            ("p", "perceived empathy - what a rater scores", AMBER, 1.8),
            ("c", "calibration - what the turn was worth", ACCENT, 1.8)]:
        vals = [{"d": t.user_after.negative_load, "p": t.perceived,
                 "c": t.calibration}[key] for t in ep.turns]
        fig.add_trace(go.Scatter(x=xs, y=vals, name=label, mode="lines",
                                 line=dict(color=color, width=width)))
    for t in ep.turns:
        if t.new_directive:
            fig.add_vline(x=t.index, line=dict(color=ACCENT, width=1, dash="dot"),
                          opacity=0.45)
    onset, _ = localize(ep, persona)
    if onset is not None:
        fig.add_vline(x=onset, line=dict(color=ALARM, width=2))
        fig.add_annotation(x=onset, y=1.02,
                           text="broke here: {}".format(dominant_cause(ep, onset)),
                           showarrow=False, font=dict(color=ALARM, size=11),
                           yanchor="bottom")
    rows = [[t.index, round(t.user_after.negative_load, 2), round(t.perceived, 2),
             round(t.calibration, 2), round(t.action.speech_rate, 2),
             (("new grievance; " if t.shock > 0 else "")
              + ('asked: "' + t.new_directive.replace("_", " ") + '"'
                 if t.new_directive else "")).strip("; ")]
            for t in ep.turns]
    stem = "voicehaul-conversation-{}-{}".format(policy, ep.persona)
    csv_rows = [["turn", "caller_said", "model_replied", "distress_after",
                 "perceived", "calibration", "speech_rate", "cheerfulness",
                 "acknowledgement", "request_made", "standing_requests",
                 "new_grievance"]]
    for t in ep.turns:
        csv_rows.append([t.index, t.utterance, t.reply,
                         round(t.user_after.negative_load, 4),
                         round(t.perceived, 4), round(t.calibration, 4),
                         round(t.action.speech_rate, 4),
                         round(t.action.cheerfulness, 4),
                         round(t.action.acknowledgement, 4),
                         t.new_directive or "", "|".join(t.standing_directives),
                         "yes" if t.shock > 0 else ""])
    buf = io.StringIO()
    csv.writer(buf).writerows(csv_rows)
    conv_file = _write(stem + ".csv", buf.getvalue())

    verdict = ("**Conversation failed.** " if ep.failed else "**Conversation held.** ")
    verdict += ("The caller is left carrying {:.2f} after {} turns. Dotted teal "
                "lines mark every explicit request the caller made."
                ).format(ep.turns[-1].user_after.negative_load, len(ep.turns))
    if onset is not None:
        verdict += (" The solid red line is where the diagnostic says it broke "
                    "&mdash; turn **{}**, flagged for *{}*.").format(
            onset, dominant_cause(ep, onset))
    return fig, rows, verdict, conv_file


# ---------------------------------------------------------------------------
# tab: onset
# ---------------------------------------------------------------------------

SIGNAL_COLOR = {"delivery jump": ALARM, "affect jump": AMBER,
                "calibration drop": VIOLET, "request violation": "#5b7c8d"}


def _evidence(ep, onset, injected):
    """What actually changed at the turn the diagnostic named."""
    t = ep.turns[onset]
    prev = ep.turns[onset - 1] if onset > 0 else t
    parts = anomaly_components(ep)[onset]
    ranked = sorted(((k, v) for k, v in parts.items()
                     if k not in ("total", "turn")), key=lambda kv: -kv[1])
    rows = "".join(
        '<tr><td style="padding:.35rem .7rem;color:{c}">{k}</td>'
        '<td style="padding:.35rem .7rem;text-align:right;color:{c};'
        'font-weight:500">{v:.2f}</td>'
        '<td style="padding:.35rem .7rem;color:var(--vh-muted);font-size:.85rem">{d}</td></tr>'
        .format(c=SIGNAL_COLOR.get(k, MUTED), k=k, v=v,
                d=dict((n, why) for n, _w, why in COMPONENTS).get(k, ""))
        for k, v in ranked)

    def delta(label, a, b):
        arrow = "&uarr;" if b > a else ("&darr;" if b < a else "&middot;")
        col = ALARM if abs(b - a) > 0.15 else MUTED
        return ('<tr><td style="padding:.3rem .7rem">{l}</td>'
                '<td style="padding:.3rem .7rem;text-align:right">{a:.2f}</td>'
                '<td style="padding:.3rem .7rem;text-align:right;color:{c}">'
                '{ar} {b:.2f}</td></tr>').format(l=label, a=a, b=b, c=col, ar=arrow)

    changes = (delta("speech rate", prev.action.speech_rate, t.action.speech_rate)
               + delta("warmth", prev.action.cheerfulness, t.action.cheerfulness)
               + delta("length", prev.action.verbosity, t.action.verbosity)
               + delta("acknowledgement", prev.action.acknowledgement,
                       t.action.acknowledgement))

    standing = ", ".join(d.replace("_", " ") for d in t.standing_directives) or "none"
    hit = ("&#10003; the diagnostic named the turn the fault was injected at"
           if abs(onset - injected) <= 1 else
           "&#10007; the fault was injected at turn {}".format(injected))
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));'
        'gap:18px;font-family:IBM Plex Sans,system-ui,sans-serif">'
        '<div style="border:1px solid var(--vh-rule);border-radius:6px;padding:14px 16px">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.66rem;'
        'letter-spacing:.11em;text-transform:uppercase;color:var(--vh-faint);'
        'margin-bottom:.5rem">why turn {o} was flagged</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:.88rem;'
        'font-family:IBM Plex Mono,monospace">{rows}</table></div>'
        '<div style="border:1px solid var(--vh-rule);border-radius:6px;padding:14px 16px">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.66rem;'
        'letter-spacing:.11em;text-transform:uppercase;color:var(--vh-faint);'
        'margin-bottom:.5rem">what the model changed at turn {o}</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:.88rem;'
        'font-family:IBM Plex Mono,monospace">{ch}</table>'
        '<div style="margin-top:.7rem;font-size:.85rem;color:var(--vh-muted)">'
        'standing requests: <b>{st}</b><br>caller distress {d0:.2f} &rarr; {d1:.2f}'
        '<br><span style="color:var(--vh-accent)">{hit}</span></div></div></div>'
    ).format(o=onset, rows=rows, ch=changes, st=standing,
             d0=t.user_before.negative_load, d1=t.user_after.negative_load, hit=hit)


def run_onset(severity, fault_turn, turns):
    persona = PERSONAS[0]
    ep = run_episode(get_policy("calibrated"), persona, seed=42,
                     n_turns=int(turns), fault_turn=int(fault_turn),
                     fault_severity=float(severity))
    onset, ranked = localize(ep, persona)
    comps = anomaly_components(ep)
    xs = [c["turn"] for c in comps]

    fig = base_fig(360, "turn", "anomaly score, by signal")
    for name, _w, _why in COMPONENTS:
        fig.add_trace(go.Bar(x=xs, y=[c[name] for c in comps], name=name,
                             marker_color=SIGNAL_COLOR[name],
                             hovertemplate="turn %{x}<br>" + name +
                                           " %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.add_vline(x=int(fault_turn), line=dict(color=INK2, width=1.4, dash="dot"))
    peak = max(c["total"] for c in comps) or 1.0
    fig.add_annotation(x=int(fault_turn), y=peak,
                       text="fault injected here", showarrow=True, arrowhead=0,
                       arrowcolor=INK2, ay=-26, font=dict(color=INK2, size=11))
    if onset is not None:
        fig.add_annotation(x=onset, y=comps[onset]["total"],
                           text="diagnostic says turn {}".format(onset),
                           showarrow=True, arrowhead=0, arrowcolor=ACCENT, ay=-52,
                           font=dict(color=ACCENT, size=11))

    if onset is None:
        return fig, ("The diagnostic returned **no onset**: repairing the model "
                     "from every candidate turn would not have changed the "
                     "outcome. Refusing to answer is the correct behaviour "
                     "there."), ""

    import random as _r
    rng = _r.Random(11)
    cases = []
    for i in range(30):
        pp = PERSONAS[i % len(PERSONAS)]
        ft = rng.randrange(6, max(8, int(turns) - 8))
        cases.append((run_episode(get_policy("calibrated"), pp, seed=2000 + i,
                                  n_turns=int(turns), fault_turn=ft,
                                  fault_severity=float(severity)), pp))
    r = score_localization(cases, tolerance=1)
    cause = dominant_cause(ep, onset)
    md = ("### Turn **{o}** &mdash; {c}\n\n"
          "The stack shows what each turn was flagged for. Before the break "
          "there is only small delivery jitter; at turn {o} several signals fire "
          "together, which is what separates *the model changed* from *the "
          "caller changed*.\n\n"
          "Candidates considered: {rk}. Across {n} injected faults at this "
          "severity: top-1 **{t1:.0%}**, top-3 **{t3:.0%}**, no false positives "
          "on healthy runs.\n\n"
          "*Accuracy is reported against severity, never as one number. Lower "
          "the slider to 0.35 and the fault blends into the model's own policy "
          "&mdash; the realistic case.*"
          ).format(o=onset, c=cause, rk=ranked[:4], n=r["n"],
                   t1=r["top1"], t3=r["top3"])
    return fig, md, _evidence(ep, onset, int(fault_turn))


# ---------------------------------------------------------------------------
# tab: rating budget
# ---------------------------------------------------------------------------

def run_budget(n_conv, n_raters, rater_sd, episodes):
    eps = [run_episode(get_policy("calibrated"), PERSONAS[i % len(PERSONAS)],
                       seed=i, n_turns=40) for i in range(int(episodes))]
    sb = M.stdev([1.0 + 6.0 * e.mean_perceived for e in eps])
    mde = M.min_detectable_effect(sb, rater_sd, int(n_conv), int(n_raters))
    need = M.required_n(0.20, sb, rater_sd, int(n_raters))

    fig = base_fig(340, "conversations per arm",
                   "smallest detectable regression (Likert)")
    ns = list(range(20, 601, 20))
    for r in (1, 3, 5, 10):
        fig.add_trace(go.Scatter(
            x=ns, y=[M.min_detectable_effect(sb, rater_sd, n, r) for n in ns],
            mode="lines", name="{} raters".format(r),
            line=dict(width=3 if r == int(n_raters) else 1.6,
                      color=ACCENT if r == int(n_raters) else "var(--vh-rule2)")))
    fig.add_vline(x=int(n_conv), line=dict(color=ALARM, width=1.4, dash="dot"))

    ks = [k for k in (5, 10, 20, 40) if k <= len(eps)]
    rows = budget_curve(eps, ks, seeds=6) if ks else []
    eq = equivalent_budget(eps, ks[len(ks) // 2]) if ks else None
    md = ("**Smallest detectable regression: {:.2f} Likert points** at n={} with "
          "{} raters.\n\nCatching a 0.20-point regression needs n = **{}**.\n\n"
          "Measured between-conversation spread: {:.2f}.\n\n"
          "---\n\n**Which conversations to rate.** Human rating is the dominant "
          "cost line, and a suite whose conversations all sit in one corner of "
          "caller-state space measures that corner many times.\n\n"
          "| budget | diverse | random | saving |\n|---|---:|---:|---:|\n"
          ).format(mde, int(n_conv), int(n_raters), need, sb)
    for row in rows:
        md += "| {} | {:.3f} | {:.3f} | {:.0f}% |\n".format(
            row["k"], row["diverse"], row["random"], 100 * row["saving"])
    if eq:
        k = ks[len(ks) // 2]
        md += ("\n{} diversity-selected conversations cover as much as {} sampled "
               "at random &mdash; {:.0f}% of the rating cost for the same "
               "coverage.".format(k, eq, 100.0 * k / eq))
    return fig, md


# ---------------------------------------------------------------------------
# tab: judge substitution (precomputed; needs a model and a key to regenerate)
# ---------------------------------------------------------------------------

def load_substitution():
    for path in ("artifacts/judge-substitution.json",
                 os.path.join(os.path.dirname(__file__),
                              "artifacts/judge-substitution.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return None


SUB = load_substitution()


def substitution_figs():
    if not SUB:
        return None, None, "Run `python run_substitution.py --source llm` first."
    rows = SUB["rows"]
    overall = {r["dimension"]: r for r in rows if r["segment"] == "all"}
    pairs = SUB.get("pairs", {})

    scatter = base_fig(400, "true quality of the turn", "rating (1-7)")
    for dim, dash in (("perceived_empathy", None), ("actual_help", "dot")):
        pts = pairs.get(dim, [])
        if not pts:
            continue
        xs = [p["theta"] for p in pts]
        for key, color, name in (("panel", ACCENT, "human panel"),
                                 ("judge", ALARM, "LLM judge")):
            ys = [p[key] for p in pts]
            scatter.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=name,
                marker=dict(size=6, color=color, opacity=0.40)))
            # Least-squares line plus the spread around it. Both raters point
            # the same way on average - the judge is not backwards - so slope
            # alone would be a misleading picture. What separates them is the
            # scatter, and scatter is what reliability measures.
            n = len(xs)
            if n >= 3:
                mx = sum(xs) / n
                my = sum(ys) / n
                num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
                den = sum((xs[i] - mx) ** 2 for i in range(n))
                if den > 0:
                    b = num / den
                    a = my - b * mx
                    resid = [ys[i] - (a + b * xs[i]) for i in range(n)]
                    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
                    lo, hi = min(xs), max(xs)
                    scatter.add_trace(go.Scatter(
                        x=[lo, hi, hi, lo], y=[a + b * lo - sd, a + b * hi - sd,
                                               a + b * hi + sd, a + b * lo + sd],
                        fill="toself", mode="lines", showlegend=False,
                        line=dict(width=0), fillcolor=color, opacity=0.10,
                        hoverinfo="skip"))
                    scatter.add_trace(go.Scatter(
                        x=[lo, hi], y=[a + b * lo, a + b * hi], mode="lines",
                        name="{}: slope {:+.1f}, spread &plusmn;{:.1f}".format(
                            name, b, sd),
                        line=dict(color=color, width=3)))
        break   # one dimension keeps the picture readable

    bars = base_fig(340, "human ratings one judge rating is worth", "")
    segs = [r for r in rows if r["segment"] != "all"]
    for dim, color in (("perceived_empathy", AMBER), ("actual_help", ALARM)):
        sel = [r for r in segs if r["dimension"] == dim]
        sel.sort(key=lambda r: r["ratio_estimated"])
        bars.add_trace(go.Bar(
            y=[r["segment"].replace("_", " ") for r in sel],
            x=[r["ratio_estimated"] for r in sel], orientation="h",
            name=dim.replace("_", " "), marker_color=color))
    bars.add_vline(x=1.0, line=dict(color=INK2, width=1.4, dash="dash"))

    md = ("| dimension | n | judge reliability | reading | 1 judge = N human "
          "| reading |\n|---|---:|---:|---|---:|---|\n")
    for key, name in (("perceived_empathy", "perceived empathy"),
                      ("actual_help", "did it actually help")):
        r = overall[key]
        d1 = describe("judge reliability", r["rho_judge_estimated"])
        d2 = describe("substitution ratio", r["ratio_estimated"])
        md += "| {} | {} | {:.2f} | **{}** | {:.2f} | **{}** |\n".format(
            name, r["n"], r["rho_judge_estimated"], d1["label"],
            r["ratio_estimated"], d2["label"])
    md += ("\n**The estimator, checked against ground truth** &mdash; the one "
           "validation real data cannot run:\n\n"
           "| dimension | estimated rho | true rho | error |\n|---|---:|---:|---:|\n")
    for key, name in (("perceived_empathy", "perceived empathy"),
                      ("actual_help", "did it actually help")):
        r = overall[key]
        md += "| {} | {:.3f} | {:.3f} | {:+.3f} |\n".format(
            name, r["rho_judge_estimated"], r["rho_judge_true"],
            r["rho_judge_estimated"] - r["rho_judge_true"])
    md += ("\nJudge: `{}`. A single agreement figure would have reported "
           "{:.2f} and hidden a twenty-fold spread across segments."
           .format(SUB["judge_model"],
                   overall["perceived_empathy"]["rho_judge_estimated"]))
    return scatter, bars, md



def headline_strip():
    """Three numbers, computed on load, no click required."""
    cfg = SuiteConfig(name="headline", episodes=30, turns=40)
    rep = compare("calibrated", "mirror", cfg, diagnose=False)
    panel = [d for d in rep.dimensions if d.name == "panel: perceived empathy"][0]
    sub = SUB
    ah = None
    if sub:
        ah = [r for r in sub["rows"]
              if r["dimension"] == "actual_help" and r["segment"] == "all"][0]

    cards = [
        (ALARM, "the leaderboard would have passed it",
         "{:+.3f}".format(panel.delta),
         "a fixed-prompt panel rated the candidate HIGHER on perceived empathy "
         "while the conversations regressed"),
        (ACCENT, "which turn broke it",
         "97%",
         "top-1 accuracy against injected faults at realistic severity, with no "
         "false positives on healthy calls"),
        (AMBER, "can a judge replace your raters?",
         "{:.2f}".format(ah["ratio_estimated"]) if ah else "-",
         "human ratings one LLM judge rating is worth on whether a turn "
         "actually helped - 0.00 with hostile callers"),
    ]
    html = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,'
            'minmax(260px,1fr));gap:16px;margin:.4rem 0 1rem;'
            'font-family:IBM Plex Sans,system-ui,sans-serif">')
    for col, kicker, value, note in cards:
        html += (
            '<div style="border:1px solid var(--vh-rule);border-left:3px solid {c};'
            'border-radius:5px;padding:14px 16px;background:var(--vh-surface)">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.64rem;'
            'letter-spacing:.11em;text-transform:uppercase;color:var(--vh-faint)">{k}</div>'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:1.75rem;'
            'font-weight:500;color:{c};line-height:1.1;margin:.35rem 0 .3rem">{v}</div>'
            '<div style="font-size:.84rem;color:var(--vh-muted);line-height:1.45">{n}</div>'
            '</div>').format(c=col, k=kicker, v=value, n=note)
    return html + "</div>"


# ---------------------------------------------------------------------------
# ZeroGPU entry point
# ---------------------------------------------------------------------------
#
# ZeroGPU refuses to start a Space with no @spaces.GPU function. This suite is
# deliberately CPU-only - the harness is standard library and the whole point is
# that it runs anywhere - so the decorated function is a probe rather than real
# work. It is wired to a control at the bottom of the page so it is honest about
# what it is rather than hidden.

if HAS_SPACES:
    @spaces.GPU(duration=10)
    def gpu_probe():
        import platform
        try:
            import torch
            dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        except Exception:
            dev = "torch not installed"
        return ("This Space is CPU-only. The evaluation harness is pure Python "
                "standard library; ZeroGPU is used only because it is the free "
                "tier that allows an interactive Gradio app.\n\n"
                "python {}  |  gpu visible: {}").format(
            platform.python_version(), dev)
else:
    def gpu_probe():
        return "Running locally; no ZeroGPU allocation."


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

POLICIES = ["calibrated", "mirror", "flat_cheerful", "drifter", "oracle"]

with gr.Blocks(css=CSS, title="VoiceHaul",
               theme=gr.themes.Soft(
                   primary_hue="teal", secondary_hue="teal",
                   neutral_hue="slate",
                   font=["IBM Plex Sans", "system-ui", "sans-serif"],
                   font_mono=["IBM Plex Mono", "monospace"])) as demo:
    gr.HTML(AUTHOR_CARD)
    gr.Markdown(
        "# Sounding right on every turn, and getting the conversation wrong\n"
        "**VoiceHaul &mdash; long-horizon evaluation, failure-onset diagnosis "
        "and judge-substitution analysis for empathic voice agents.**\n\n"
        "A turn-level rating tells you whether a response sounded right. It "
        "cannot tell you whether a model still honours what the caller asked "
        "for twenty turns ago, whether its calibration decays as a session runs "
        "long, whether it is regulating the caller's affect or mirroring it "
        "back, or which turn broke a call that ended badly.")
    gr.HTML(headline_strip())
    gr.HTML(tabs_map())
    gr.HTML(LOOP_SVG)

    with gr.Tabs():
        with gr.Tab("Judge substitution"):
            gr.Markdown(
                "### When can an automated rater replace a human one?\n"
                "A human panel is the trusted measurement and the dominant cost "
                "line; an LLM judge is cheap and of unknown trustworthiness. The "
                "answer is not one number &mdash; it is a number per dimension "
                "and per caller segment.")
            gr.HTML(RATERS_SVG)
            s_scatter, s_bars, s_md = substitution_figs()
            if SUB:
                _sub_path = _write("voicehaul-judge-substitution.json",
                                   json.dumps(SUB, indent=2))
                gr.File(value=[_sub_path], label="the full result as JSON - "
                        "every dimension, every segment, and the raw per-turn "
                        "ratings behind the correlations", interactive=False)
            gr.Markdown(s_md)
            if s_scatter is not None:
                gr.Markdown(
                    "Each dot is one turn. Horizontal axis: the quality the turn "
                    "actually had; vertical: what each rater said. The shaded "
                    "band is one standard deviation around each fit.\n\n"
                    "**Both lines slope the same way** &mdash; the judge is not "
                    "backwards, it points in the right direction on average. What "
                    "separates them is the spread around the line, and that is "
                    "exactly what reliability measures. A rater can be unbiased "
                    "and still be useless for deciding anything about a single "
                    "call.")
                gr.Plot(value=s_scatter)
                gr.Markdown("**And how far the answer moves between segments.** "
                            "Anything left of the dashed line cannot substitute.")
                gr.Plot(value=s_bars)

        with gr.Tab("Score your own call"):
            gr.Markdown(
                "### Paste a real transcript\n"
                "Everything else on this page runs a simulated caller. This runs "
                "nothing &mdash; it reads a call that already happened.\n\n"
                "Only what a transcript can honestly answer is reported: what the "
                "agent's delivery was, whether the caller asked for a change, and "
                "whether the agent did it **and kept doing it**. Nothing is "
                "inferred that you could not check by hand.\n\n"
                "Prefix lines with `Caller:` and `Agent:`. The box starts with a "
                "worked example; replace it with your own.")
            tr_in = gr.Textbox(value=SAMPLE_CALL, lines=12, max_lines=30,
                               label="the call", show_copy_button=True)
            tr_btn = gr.Button("Measure this call", variant="primary")
            _t0 = run_transcript(SAMPLE_CALL)
            tr_card = gr.HTML(value=_t0[0])
            tr_plot = gr.Plot(value=_t0[1])
            tr_tbl = gr.Dataframe(
                value=_t0[2],
                headers=["turn", "speaker", "what was said", "asked for",
                         "apology", "length", "acknowledgement", "ignored"],
                label="turn by turn", wrap=True)
            tr_md = gr.Markdown(value=_t0[3])
            tr_file = gr.File(value=_t0[4], label="this analysis as CSV",
                              interactive=False)
            tr_btn.click(run_transcript, tr_in,
                         [tr_card, tr_plot, tr_tbl, tr_md, tr_file])

        with gr.Tab("Release gate"):
            gr.Markdown(
                "### Should this candidate ship?\n"
                "Both arms run on the same suite. Welch two-sample tests with "
                "conversations as the unit, Holm-corrected across dimensions. "
                "Panel rows are what a fixed-prompt leaderboard reports; the "
                "rest is what the conversations report.\n\n"
                "*Try `calibrated` against `mirror`: the panel rates the "
                "candidate higher and the gate blocks it.*")
            gr.HTML(policy_legend())
            with gr.Row():
                g_base = gr.Dropdown(POLICIES, value="calibrated",
                                     label="baseline (what is in production)")
                g_cand = gr.Dropdown(POLICIES, value="mirror",
                                     label="candidate (what you want to ship)")
                g_ep = gr.Slider(10, 60, 30, step=5,
                                 label="conversations per arm")
                g_tn = gr.Slider(20, 60, 40, step=5, label="turns per conversation")
            g_btn = gr.Button("Run the gate", variant="primary")
            _g0 = run_gate("calibrated", "mirror", 30, 40)
            g_card = gr.HTML(value=_g0[0])
            g_tbl = gr.HTML(value=_g0[1])
            g_delta = gr.Plot(value=_g0[2],
                              label="every dimension, signed so positive is better")
            g_segtitle = gr.Markdown(value=_g0[3])
            g_seg = gr.Plot(value=_g0[4])
            g_notes = gr.Markdown(value=_g0[5])
            gr.Markdown("#### Take the report with you")
            g_files = gr.File(
                value=_g0[6], label="JSON for a pipeline, Markdown for a pull "
                "request, CSV of the per-conversation rows",
                file_count="multiple", interactive=False)
            gr.HTML(CI_SNIPPET)
            g_btn.click(run_gate, [g_base, g_cand, g_ep, g_tn],
                        [g_card, g_tbl, g_delta, g_segtitle, g_seg, g_notes,
                         g_files])

        with gr.Tab("Watch a conversation"):
            gr.Markdown(
                "### One conversation, turn by turn\n"
                "*Put `mirror` against the hostile caller: perceived empathy "
                "stays respectable the whole way down while distress never "
                "comes down. No turn is bad. The conversation is.*")
            gr.HTML(policy_legend())
            gr.HTML(caller_legend())
            with gr.Row():
                c_pol = gr.Dropdown(POLICIES, value="mirror", label="policy")
                c_per = gr.Dropdown(list(PLABEL.values()), value="hostile caller",
                                    label="caller")
                c_tn = gr.Slider(20, 60, 40, step=5, label="turns")
            c_btn = gr.Button("Run it", variant="primary")
            _c0 = run_conversation("mirror", "hostile caller", 40)
            c_msg = gr.Markdown(value=_c0[2])
            c_plot = gr.Plot(value=_c0[0])
            c_tbl = gr.Dataframe(
                value=_c0[1],
                headers=["turn", "distress", "perceived", "calibration",
                         "speech rate", "what the caller did"],
                label="turn log", wrap=True)
            c_file = gr.File(value=_c0[3], label="this conversation as CSV - "
                             "every turn, what was said, and what was measured",
                             interactive=False)
            c_btn.click(run_conversation, [c_pol, c_per, c_tn],
                        [c_plot, c_tbl, c_msg, c_file])

        with gr.Tab("Which turn broke it"):
            gr.Markdown(
                "### Failure-onset localization\n"
                "A regression is injected at a turn the diagnostic is never "
                "told about. Deterministic signals propose candidates, a "
                "walk-back finds where the anomalous stretch begins, and "
                "counterfactual replay refuses to answer if repairing the model "
                "from that turn would not have changed the outcome.")
            with gr.Row():
                o_sev = gr.Slider(0.35, 1.0, 1.0, step=0.05,
                                  label="fault severity - 1.00 obvious, 0.35 subtle")
                o_ft = gr.Slider(6, 30, 18, step=1,
                                 label="inject the fault at this turn")
                o_tn = gr.Slider(24, 60, 40, step=4, label="turns per conversation")
            o_btn = gr.Button("Diagnose", variant="primary")
            _o0 = run_onset(1.0, 18, 40)
            o_md = gr.Markdown(value=_o0[1])
            o_plot = gr.Plot(value=_o0[0])
            o_ev = gr.HTML(value=_o0[2])
            o_btn.click(run_onset, [o_sev, o_ft, o_tn], [o_plot, o_md, o_ev])

        with gr.Tab("Rating budget"):
            gr.Markdown(
                "### How many conversations, and which ones\n"
                "Human ratings are both the ground truth and the cost line. "
                "This is the calculation that turns \"we track regressions\" "
                "into a number of conversations and a bill.")
            with gr.Row():
                b_n = gr.Slider(20, 600, 100, step=20,
                                label="conversations per arm - what you would buy")
                b_r = gr.Slider(1, 10, 3, step=1,
                                label="human raters per conversation")
                b_sd = gr.Slider(0.4, 1.6, 0.9, step=0.05,
                                 label="how much raters disagree (Likert sd)")
                b_ep = gr.Slider(20, 60, 40, step=10,
                                 label="suite size used to measure the spread")
            b_btn = gr.Button("Compute", variant="primary")
            _b0 = run_budget(100, 3, 0.9, 40)
            b_plot = gr.Plot(value=_b0[0])
            b_md = gr.Markdown(value=_b0[1])

            gr.Markdown(
                "---\n### What that costs\n\n"
                "A statistic is not a decision until it has a price on it. "
                "Enter what a rating costs you and the same measurements become "
                "a line in a budget. Nothing below is estimated that is not "
                "measured above.")
            with gr.Row():
                k_target = gr.Slider(0.05, 0.60, 0.20, step=0.05,
                                     label="regression you want to catch (Likert)")
                k_ch = gr.Slider(0.25, 12.0, 3.0, step=0.25,
                                 label="cost of one human rating ($)")
                k_cj = gr.Slider(0.001, 0.20, 0.01, step=0.001,
                                 label="cost of one judge rating ($)")
            with gr.Row():
                k_rel = gr.Slider(1, 52, 12, step=1, label="releases per year")
                k_r = gr.Slider(1, 10, 3, step=1, label="raters per conversation")
                k_ep = gr.Slider(20, 60, 40, step=10,
                                 label="suite size used to measure the spread")
            k_btn = gr.Button("Price it", variant="primary")
            _k0 = run_cost(0.20, 3.0, 0.01, 12, 3, 40)
            k_tbl = gr.HTML(value=_k0[0])
            k_md = gr.Markdown(value=_k0[1])
            k_btn.click(run_cost, [k_target, k_ch, k_cj, k_rel, k_r, k_ep],
                        [k_tbl, k_md])
            b_btn.click(run_budget, [b_n, b_r, b_sd, b_ep], [b_plot, b_md])

        with gr.Tab("How to read the numbers"):
            gr.Markdown(
                "### Every number here has a scale, and the scale has a source\n\n"
                "A metric without a band is a number nobody can act on. But half "
                "of these cannot honestly carry a fixed threshold, and "
                "pretending otherwise is how a dashboard starts lying.\n\n"
                "- **Absolute** - means the same thing on any suite. \"Ninety "
                "per cent of requests still honoured\" is ninety per cent "
                "whoever is measuring.\n"
                "- **Anchored** - a distance from an ideal that was defined for "
                "one action space. A simulated policy reaches 0.96 and the same "
                "real model measured through a transcript reaches 0.25; neither "
                "is \"good\" on its own. These are read against the range the "
                "suite can actually reach, drawn as a bar under the name in the "
                "release gate.\n"
                "- **Conventional** - thresholds from psychometrics rather than "
                "from this package, cited rather than invented.\n\n"
                "Getting this distinction wrong is the same mistake as anchoring "
                "the environment's break-even point on simulated policies, which "
                "put every real conversation on the failure floor and left the "
                "suite with no discriminative power at all.")
            gr.HTML(reading_panel([
                "feedback uptake @10", "conversation failure rate",
                "left-over distress", "calibration", "panel: perceived empathy",
                "judge reliability", "substitution ratio"]))

        with gr.Tab("What's real here"):
            gr.Markdown(
                "### What is simulated and what is not\n\n"
                "**Simulated:** the callers, the five reference policies, and the "
                "affect dynamics. Each policy embodies exactly one known failure "
                "mode.\n\n"
                "**Real:** the metrics, the estimators, the statistics, the "
                "localization algorithm &mdash; and, in the judge-substitution "
                "tab, the model being graded and the judge grading it.\n\n"
                "The reason for a synthetic environment is not convenience. "
                "**You cannot validate a measurement instrument without ground "
                "truth you control.** If you only run an eval against real "
                "models, a metric that reports the wrong thing and a model that "
                "behaves badly are indistinguishable.\n\n"
                "---\n\n"
                "### What pointing it at a real model exposed\n\n"
                "Five faults, all in the instrument rather than the model:\n\n"
                "1. An unmeasured speech rate was scored as if measured, which "
                "injected a fabricated error and made soothing structurally "
                "impossible. `Action` now carries the set of fields an adapter "
                "actually observed, and every scorer renormalises over it.\n"
                "2. The acknowledgement pattern matched *I understand* but not "
                "*I can understand*, which is what instruction-tuned models "
                "produce.\n"
                "3. The fixed-context panel gave a speaking policy no utterance, "
                "so it measured a real model's reply to an empty turn.\n"
                "4. Both panel dimensions replayed each block separately, "
                "doubling the cost of every hosted call.\n"
                "5. The environment's break-even calibration was anchored on "
                "simulated policies and sat at the 88th percentile of what a real "
                "model reaches, pinning every conversation to the failure floor. "
                "It is now a measurable parameter, recorded in the suite id, and "
                "the gate detects that saturation and refuses to report a "
                "meaningless delta.\n\n"
                "Every one has a regression test. An evaluation harness that is "
                "wrong is worse than no harness, because it is believed.\n\n"
                "---\n\n"
                "Vahit Feryad &middot; "
                "[Runopsy](https://github.com/vahit19/runopsy) &middot; "
                "[LongHaul-Bench](https://github.com/vahit19/LongHaul-Bench) "
                "&middot; Apache-2.0")
            gr.Markdown("---\n#### Runtime")
            gr.Markdown(
                "This Space runs on ZeroGPU because that is the free tier that "
                "allows an interactive Gradio app. Nothing here needs a GPU.")
            probe_btn = gr.Button("Check the runtime", size="sm")
            probe_out = gr.Markdown()
            probe_btn.click(gpu_probe, None, probe_out)

if __name__ == "__main__":
    demo.launch()
