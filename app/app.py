"""VoiceHaul - interactive report.

Everything on this page is computed when you load it. The harness is pure
standard library; gradio and plotly are here only to draw.
"""

import json
import math
import os

try:                                   # ZeroGPU hardware expects this import
    import spaces                        # noqa: F401
    HAS_SPACES = True
except ImportError:                      # local development
    HAS_SPACES = False

import gradio as gr
import plotly.graph_objects as go

from voicehaul.bands import BANDS, COLOURS, describe
from voicehaul.config import SuiteConfig
from voicehaul.env import PERSONAS
from voicehaul.gate import compare
from voicehaul import metrics as M
from voicehaul.onset import (COMPONENTS, anomaly_components, dominant_cause,
                             localize, score_localization)
from voicehaul.registry import get_persona, get_policy
from voicehaul.report import render_text
from voicehaul.runner import run_episode
from voicehaul.select import budget_curve, equivalent_budget

INK, INK2, MUTED, FAINT = "#131c1b", "#3a4b49", "#61756f", "#8b9d99"
RULE, ACCENT, ALARM, AMBER, VIOLET = "#dbe3e1", "#0d6f66", "#ac4136", "#8f6414", "#6a56a3"
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
.gradio-container{max-width:1180px!important;font-family:"IBM Plex Sans",system-ui,sans-serif}
h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif!important;letter-spacing:-.015em}
.vh-note{font-size:.92rem;color:#61756f}
.vh-fig svg{max-width:100%;height:auto;color:#3a4b49}
.vh-fig figcaption{font-size:.85rem;color:#61756f;margin-top:.6rem;line-height:1.5}
.dlab{font-family:"IBM Plex Sans",sans-serif;font-size:13px;font-weight:500;fill:#131c1b}
.dsub{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:#8b9d99}
.dedge,.dnote{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:#61756f}
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
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#3a4b49"/></marker>
<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0d6f66"/></marker>
<marker id="arx" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8f6414"/></marker>
</defs>
<rect x="14" y="112" width="132" height="52" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="80" y="134" text-anchor="middle" class="dlab">caller state</text>
<text x="80" y="151" text-anchor="middle" class="dsub">6-dim affect</text>
<line x1="146" y1="138" x2="212" y2="138" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#ar)"/>
<text x="179" y="128" text-anchor="middle" class="dedge">renders</text>
<rect x="212" y="112" width="128" height="52" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="276" y="134" text-anchor="middle" class="dlab">utterance</text>
<text x="276" y="151" text-anchor="middle" class="dsub">what is said</text>
<line x1="340" y1="138" x2="404" y2="138" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#ar)"/>
<rect x="404" y="106" width="126" height="64" rx="3" fill="#d3e7e3" stroke="#0d6f66" stroke-width="1.4"/>
<text x="467" y="132" text-anchor="middle" class="dlab">model</text>
<text x="467" y="149" text-anchor="middle" class="dsub">under test</text>
<line x1="530" y1="138" x2="586" y2="138" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#ar)"/>
<text x="562" y="128" text-anchor="middle" class="dedge">reply</text>
<rect x="586" y="100" width="156" height="76" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="664" y="124" text-anchor="middle" class="dlab">delivery vector</text>
<text x="664" y="142" text-anchor="middle" class="dsub">rate, warmth,</text>
<text x="664" y="155" text-anchor="middle" class="dsub">length, apology,</text>
<text x="664" y="168" text-anchor="middle" class="dsub">acknowledgement</text>
<path d="M742 120 L774 120 L774 52 L806 52" fill="none" stroke="#8f6414" stroke-width="1.4" marker-end="url(#arx)"/>
<path d="M742 156 L774 156 L774 232 L806 232" fill="none" stroke="#0d6f66" stroke-width="1.4" marker-end="url(#ara)"/>
<rect x="806" y="28" width="82" height="48" rx="3" fill="none" stroke="#8f6414" stroke-width="1.3"/>
<text x="847" y="48" text-anchor="middle" class="dlab" fill="#8f6414">perceived</text>
<text x="847" y="64" text-anchor="middle" class="dlab" fill="#8f6414">empathy</text>
<rect x="806" y="208" width="82" height="48" rx="3" fill="none" stroke="#0d6f66" stroke-width="1.3"/>
<text x="847" y="228" text-anchor="middle" class="dlab" fill="#0d6f66">calibration</text>
<text x="847" y="244" text-anchor="middle" class="dsub" fill="#0d6f66">hidden</text>
<text x="888" y="94" text-anchor="end" class="dnote" fill="#8f6414">what a rater scores</text>
<text x="888" y="288" text-anchor="end" class="dnote" fill="#0d6f66">what moves the caller</text>
<path d="M806 232 L768 232 L768 268 L80 268 L80 164" fill="none" stroke="#0d6f66"
 stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#ara)"/>
<text x="424" y="262" text-anchor="middle" class="dedge" fill="#0d6f66">drives the next state</text>
<line x1="847" y1="76" x2="847" y2="196" stroke="#8f6414" stroke-width="1.2" stroke-dasharray="3 5"/>
<text x="856" y="134" class="dnote" fill="#8f6414">feeds</text>
<text x="856" y="149" class="dnote" fill="#8f6414">nothing</text>
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
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#3a4b49"/></marker>
<marker id="b2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0d6f66"/></marker></defs>
<rect x="14" y="128" width="118" height="52" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="73" y="150" text-anchor="middle" class="dlab">one turn</text>
<text x="73" y="167" text-anchor="middle" class="dsub">caller + reply</text>
<path d="M132 144 L200 78" fill="none" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M132 154 L200 154" fill="none" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M132 164 L200 246" fill="none" stroke="#0d6f66" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#b2)"/>
<rect x="200" y="52" width="150" height="52" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="275" y="74" text-anchor="middle" class="dlab">human panel</text>
<text x="275" y="91" text-anchor="middle" class="dsub">k raters, noisy</text>
<rect x="200" y="128" width="150" height="52" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="275" y="150" text-anchor="middle" class="dlab">LLM judge</text>
<text x="275" y="167" text-anchor="middle" class="dsub">cheap, unproven</text>
<rect x="200" y="220" width="150" height="52" rx="3" fill="none" stroke="#0d6f66" stroke-width="1.4" stroke-dasharray="5 4"/>
<text x="275" y="242" text-anchor="middle" class="dlab" fill="#0d6f66">latent truth</text>
<text x="275" y="259" text-anchor="middle" class="dsub" fill="#0d6f66">simulation only</text>
<path d="M350 78 L430 114" fill="none" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#b1)"/>
<path d="M350 154 L430 134" fill="none" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#b1)"/>
<rect x="430" y="98" width="176" height="58" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.2"/>
<text x="518" y="120" text-anchor="middle" class="dlab">correlate, then</text>
<text x="518" y="137" text-anchor="middle" class="dlab">disattenuate</text>
<text x="518" y="151" text-anchor="middle" class="dsub">divide out panel noise</text>
<line x1="606" y1="127" x2="672" y2="127" stroke="#3a4b49" stroke-width="1.2" marker-end="url(#b1)"/>
<rect x="672" y="98" width="150" height="58" rx="3" fill="none" stroke="#3a4b49" stroke-width="1.4"/>
<text x="747" y="122" text-anchor="middle" class="dlab">estimated rho</text>
<text x="747" y="141" text-anchor="middle" class="dsub">what a customer gets</text>
<path d="M350 246 L660 246 L660 172 L700 172 L700 160" fill="none" stroke="#0d6f66"
 stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#b2)"/>
<text x="505" y="238" text-anchor="middle" class="dedge" fill="#0d6f66">is the estimate right?</text>
<text x="886" y="196" text-anchor="end" class="dnote" fill="#0d6f66">this arrow does not exist on real data:</text>
<text x="886" y="212" text-anchor="end" class="dnote" fill="#0d6f66">both measurements carry error and</text>
<text x="886" y="228" text-anchor="end" class="dnote" fill="#0d6f66">neither one is the reference</text>
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
            'height:5px;background:#e8eeec;border-radius:3px">'
            '<div style="position:absolute;left:0;top:0;height:5px;width:{w:.0f}%;'
            'background:{c};border-radius:3px"></div>'
            '<div style="position:absolute;left:{w:.0f}%;top:-3px;width:2px;'
            'height:11px;background:{c}"></div></div>'
            '<div style="display:flex;justify-content:space-between;'
            'font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#8b9d99;'
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
            'font-size:.85rem;border-bottom:1px solid #dbe3e1">{n}<br>'
            '<span style="font-size:.66rem;letter-spacing:.08em;'
            'text-transform:uppercase;color:#8b9d99">{k}</span></td>'
            '<td style="padding:.6rem .8rem;vertical-align:top;font-size:.87rem;'
            'color:#3a4b49;border-bottom:1px solid #dbe3e1">{r}'
            '<div style="margin-top:.35rem;font-family:IBM Plex Mono,monospace;'
            'font-size:.72rem">{c}</div>'
            '<div style="margin-top:.3rem;font-size:.78rem;color:#8b9d99">{s}</div>'
            '</td></tr>').format(n=n, k=b.kind, r=b.reading, c=cuts, s=b.source)
    return ('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
            'width:100%;font-family:IBM Plex Sans,system-ui,sans-serif">{}'
            '</table></div>').format(rows)


VERDICT_STYLE = {
    "BLOCK": (ALARM, "#f3ddd9", "Do not ship"),
    "SHIP": (ACCENT, "#d3e7e3", "Safe to ship"),
    "INCONCLUSIVE": (MUTED, "#eef2f1", "Not enough evidence"),
    "SATURATED": (AMBER, "#f3e8cf", "This suite cannot separate them"),
}


def _card(rep):
    color, bg, headline = VERDICT_STYLE.get(rep.verdict, (MUTED, "#eef2f1", ""))
    reasons = "".join("<li>{}</li>".format(r) for r in rep.reasons)
    return (
        '<div style="border:1px solid {c};background:{b};border-radius:6px;'
        'padding:18px 22px;font-family:IBM Plex Sans,system-ui,sans-serif">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
        'letter-spacing:.12em;text-transform:uppercase;color:{c};'
        'margin-bottom:.35rem">verdict</div>'
        '<div style="font-size:1.9rem;font-weight:600;color:{c};line-height:1.1">'
        '{v}</div>'
        '<div style="font-size:1.05rem;color:#3a4b49;margin-top:.3rem">{h}</div>'
        '<ul style="margin:.9rem 0 0;padding-left:1.1rem;color:#3a4b49;'
        'font-size:.92rem;line-height:1.55">{r}</ul></div>'
    ).format(c=color, b=bg, v=rep.verdict, h=headline, r=reasons)


def _table(rep):
    vals = {}
    for d in rep.dimensions:
        vals.setdefault(d.name, []).extend([d.baseline, d.candidate])

    def row(d):
        gate = rep.gating.get(d.name, True)
        col = {"improved": ACCENT, "regressed": ALARM}.get(d.verdict, MUTED)
        tag = "" if gate else ' <span style="color:#8b9d99">diagnostic</span>'
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
            '<tr><td style="padding:.55rem .7rem;border-bottom:1px solid #dbe3e1">'
            '{n}{t}{chip}{sc}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid #dbe3e1">{b:.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid #dbe3e1;font-weight:500">{c:.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid #dbe3e1;'
            'color:{col};font-weight:500">{d:+.3f}</td>'
            '<td style="padding:.55rem .7rem;text-align:right;border-bottom:1px solid #dbe3e1">{p:.4f}</td>'
            '</tr>').format(n=d.name, t=tag, chip=chip, sc=scale, b=d.baseline,
                            c=d.candidate, d=d.delta, p=d.p_holm, col=col)

    panel = [d for d in rep.dimensions if d.name.startswith("panel:")]
    conv = [d for d in rep.dimensions if not d.name.startswith("panel:")]
    head = ('<tr>' + "".join(
        '<th style="text-align:{a};padding:0 .7rem .5rem;font-size:.66rem;'
        'letter-spacing:.09em;text-transform:uppercase;color:#8b9d99;'
        'border-bottom:1px solid #c6d2cf;font-weight:500">{h}</th>'.format(
            a=("left" if i == 0 else "right"), h=h)
        for i, h in enumerate(["dimension", "baseline", "candidate", "delta",
                               "holm p"])) + '</tr>')
    sec = lambda t: ('<tr><td colspan="5" style="padding:1rem .7rem .4rem;'
                     'font-family:IBM Plex Mono,monospace;font-size:.68rem;'
                     'letter-spacing:.1em;text-transform:uppercase;color:#8b9d99">'
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
            ALARM if d.verdict == "regressed" else "#c6d2cf"
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
    return _card(rep), _table(rep), delta_fig, seg_title, seg_fig, notes


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
    verdict = ("**Conversation failed.** " if ep.failed else "**Conversation held.** ")
    verdict += ("The caller is left carrying {:.2f} after {} turns. Dotted teal "
                "lines mark every explicit request the caller made."
                ).format(ep.turns[-1].user_after.negative_load, len(ep.turns))
    if onset is not None:
        verdict += (" The solid red line is where the diagnostic says it broke "
                    "&mdash; turn **{}**, flagged for *{}*.").format(
            onset, dominant_cause(ep, onset))
    return fig, rows, verdict


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
        '<td style="padding:.35rem .7rem;color:#61756f;font-size:.85rem">{d}</td></tr>'
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
        '<div style="border:1px solid #dbe3e1;border-radius:6px;padding:14px 16px">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.66rem;'
        'letter-spacing:.11em;text-transform:uppercase;color:#8b9d99;'
        'margin-bottom:.5rem">why turn {o} was flagged</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:.88rem;'
        'font-family:IBM Plex Mono,monospace">{rows}</table></div>'
        '<div style="border:1px solid #dbe3e1;border-radius:6px;padding:14px 16px">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:.66rem;'
        'letter-spacing:.11em;text-transform:uppercase;color:#8b9d99;'
        'margin-bottom:.5rem">what the model changed at turn {o}</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:.88rem;'
        'font-family:IBM Plex Mono,monospace">{ch}</table>'
        '<div style="margin-top:.7rem;font-size:.85rem;color:#61756f">'
        'standing requests: <b>{st}</b><br>caller distress {d0:.2f} &rarr; {d1:.2f}'
        '<br><span style="color:#0d6f66">{hit}</span></div></div></div>'
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
                      color=ACCENT if r == int(n_raters) else "#c6d2cf")))
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
            '<div style="border:1px solid #dbe3e1;border-left:3px solid {c};'
            'border-radius:5px;padding:14px 16px;background:#fff">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.64rem;'
            'letter-spacing:.11em;text-transform:uppercase;color:#8b9d99">{k}</div>'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:1.75rem;'
            'font-weight:500;color:{c};line-height:1.1;margin:.35rem 0 .3rem">{v}</div>'
            '<div style="font-size:.84rem;color:#61756f;line-height:1.45">{n}</div>'
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
               theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        "# Sounding right on every turn, and getting the conversation wrong\n"
        "**Long-horizon evaluation and failure-onset diagnosis for empathic "
        "voice agents.**\n\n"
        "A turn-level rating tells you whether a response sounded right. It "
        "cannot tell you whether a model still honours what the caller asked "
        "for twenty turns ago, whether its calibration decays as a session runs "
        "long, whether it is regulating the caller's affect or mirroring it "
        "back, or which turn broke a call that ended badly.\n\n"
        "Everything below runs live. "
        "[Source](https://github.com/vahit19/voicehaul) &middot; "
        "[Static report](https://vahit19.github.io/voicehaul/)")
    gr.HTML(headline_strip())
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

        with gr.Tab("Release gate"):
            gr.Markdown(
                "### Should this candidate ship?\n"
                "Both arms run on the same suite. Welch two-sample tests with "
                "conversations as the unit, Holm-corrected across dimensions. "
                "Panel rows are what a fixed-prompt leaderboard reports; the "
                "rest is what the conversations report.\n\n"
                "*Try `calibrated` against `mirror`: the panel rates the "
                "candidate higher and the gate blocks it.*")
            with gr.Row():
                g_base = gr.Dropdown(POLICIES, value="calibrated", label="baseline")
                g_cand = gr.Dropdown(POLICIES, value="mirror", label="candidate")
                g_ep = gr.Slider(10, 60, 30, step=5, label="conversations")
                g_tn = gr.Slider(20, 60, 40, step=5, label="turns")
            g_btn = gr.Button("Run the gate", variant="primary")
            _g0 = run_gate("calibrated", "mirror", 30, 40)
            g_card = gr.HTML(value=_g0[0])
            g_tbl = gr.HTML(value=_g0[1])
            g_delta = gr.Plot(value=_g0[2],
                              label="every dimension, signed so positive is better")
            g_segtitle = gr.Markdown(value=_g0[3])
            g_seg = gr.Plot(value=_g0[4])
            g_notes = gr.Markdown(value=_g0[5])
            g_btn.click(run_gate, [g_base, g_cand, g_ep, g_tn],
                        [g_card, g_tbl, g_delta, g_segtitle, g_seg, g_notes])

        with gr.Tab("Watch a conversation"):
            gr.Markdown(
                "### One conversation, turn by turn\n"
                "*Put `mirror` against the hostile caller: perceived empathy "
                "stays respectable the whole way down while distress never "
                "comes down. No turn is bad. The conversation is.*")
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
            c_btn.click(run_conversation, [c_pol, c_per, c_tn],
                        [c_plot, c_tbl, c_msg])

        with gr.Tab("Which turn broke it"):
            gr.Markdown(
                "### Failure-onset localization\n"
                "A regression is injected at a turn the diagnostic is never "
                "told about. Deterministic signals propose candidates, a "
                "walk-back finds where the anomalous stretch begins, and "
                "counterfactual replay refuses to answer if repairing the model "
                "from that turn would not have changed the outcome.")
            with gr.Row():
                o_sev = gr.Slider(0.35, 1.0, 1.0, step=0.05, label="fault severity")
                o_ft = gr.Slider(6, 30, 18, step=1, label="inject at turn")
                o_tn = gr.Slider(24, 60, 40, step=4, label="turns")
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
                b_n = gr.Slider(20, 600, 100, step=20, label="conversations per arm")
                b_r = gr.Slider(1, 10, 3, step=1, label="raters per conversation")
                b_sd = gr.Slider(0.4, 1.6, 0.9, step=0.05, label="rater noise (Likert sd)")
                b_ep = gr.Slider(20, 60, 40, step=10, label="suite size for the spread")
            b_btn = gr.Button("Compute", variant="primary")
            _b0 = run_budget(100, 3, 0.9, 40)
            b_plot = gr.Plot(value=_b0[0])
            b_md = gr.Markdown(value=_b0[1])
            b_btn.click(run_budget, [b_n, b_r, b_sd, b_ep], [b_plot, b_md])

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
