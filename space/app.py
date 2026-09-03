"""VoiceHaul - long-horizon evaluation for empathic voice agents.

Everything on this page is computed live when you load it. Change the suite
parameters in the sidebar and every table and chart recomputes.
"""

import math

import plotly.graph_objects as go
import streamlit as st

from voicehaul.agents import (CalibratedAgent, DrifterAgent, FlatAgent,
                              MirrorAgent, OracleAgent)
from voicehaul.env import PERSONAS
from voicehaul import metrics as M
from voicehaul.onset import anomaly_scores, false_positive_rate, localize, score_localization
from voicehaul.runner import run_episode

st.set_page_config(page_title="VoiceHaul", page_icon="🎚️", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

INK, INK2, MUTED, FAINT = "#131c1b", "#3a4b49", "#61756f", "#8b9d99"
GROUND, SURFACE, RULE = "#eff2f1", "#ffffff", "#dbe3e1"
ACCENT, ACCENT_SOFT = "#0d6f66", "#d3e7e3"
ALARM, ALARM_SOFT = "#ac4136", "#f3ddd9"
AMBER, AMBER_SOFT = "#8f6414", "#f3e8cf"
VIOLET = "#6a56a3"

AGENT_COLOR = {"mirror": ALARM, "flat_cheerful": AMBER, "drifter": VIOLET,
               "calibrated": ACCENT, "oracle": "#93a5a1"}
AGENT_LABEL = {"mirror": "mirror", "flat_cheerful": "flat", "drifter": "drifter",
               "calibrated": "calibrated", "oracle": "oracle"}
AGENT_BLURB = {
    "mirror": "Matches the caller's energy and mood. Sounds deeply attuned.",
    "flat_cheerful": "Constant upbeat persona. Ignores affect and ignores requests.",
    "drifter": "Well calibrated early, degrades as the session runs long.",
    "calibrated": "Tracks affect, steers toward calm, holds requests indefinitely.",
    "oracle": "Noiseless upper bound. The ceiling every metric is read against.",
}
PERSONA_LABEL = {"distressed_billing": "billing dispute",
                 "hostile_escalation": "hostile caller",
                 "confused_elderly": "confused caller",
                 "grieving_claim": "grieving claimant",
                 "cautious_optimist": "calm caller"}
AGENTS = [("mirror", MirrorAgent), ("flat_cheerful", FlatAgent),
          ("drifter", DrifterAgent), ("calibrated", CalibratedAgent),
          ("oracle", OracleAgent)]

st.markdown("""<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
.stApp { background: #eff2f1; }
html, body, [class*="css"], .stMarkdown, p, li, div[data-testid="stMarkdownContainer"] {
  font-family: "IBM Plex Sans", system-ui, sans-serif; color: #131c1b; }
h1, h2, h3 { font-family: "IBM Plex Serif", Georgia, serif !important;
  letter-spacing: -.015em; color: #131c1b !important; }
h1 { font-size: 2.6rem !important; line-height: 1.06 !important; font-weight: 600 !important; }
h2 { font-size: 1.6rem !important; font-weight: 600 !important; margin-top: .3rem !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
.block-container { padding-top: 3.6rem; padding-bottom: 5rem; max-width: 1280px; }
.vh-kicker { font-family: "IBM Plex Mono", monospace; font-size: .72rem;
  letter-spacing: .14em; text-transform: uppercase; color: #8b9d99; margin-bottom: .5rem; }
.vh-lede { font-size: 1.12rem; line-height: 1.6; color: #3a4b49; max-width: 60ch; }
.vh-note { font-size: .9rem; color: #61756f; }
.vh-prose { max-width: 68ch; }
.vh-cards { display: flex; flex-wrap: wrap; gap: 0; border: 1px solid #dbe3e1;
  border-radius: 4px; overflow: hidden; background: #fff; margin: 1.4rem 0 .6rem; }
.vh-cards > div { flex: 1 1 190px; padding: 1rem 1.15rem; border-right: 1px solid #dbe3e1; }
.vh-cards > div:last-child { border-right: 0; }
.vh-cards .k { font-family: "IBM Plex Mono", monospace; font-size: .66rem;
  letter-spacing: .12em; text-transform: uppercase; color: #8b9d99; margin-bottom: .35rem; }
.vh-cards .v { font-family: "IBM Plex Mono", monospace; font-size: 1.65rem;
  font-weight: 500; line-height: 1; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.vh-cards .s { font-size: .8rem; color: #61756f; margin-top: .4rem; line-height: 1.4; }
.vh-panel { background: #fff; border: 1px solid #dbe3e1; border-radius: 4px;
  padding: 1.1rem 1.25rem; margin: .6rem 0 1rem; }
.vh-callout { border-left: 2px solid #0d6f66; padding: .1rem 0 .1rem 1rem;
  margin: 1rem 0; color: #3a4b49; max-width: 68ch; }
.vh-callout.warn { border-left-color: #8f6414; }
table.vh { border-collapse: collapse; width: 100%; font-size: .86rem;
  font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
table.vh th { font-size: .64rem; font-weight: 500; letter-spacing: .09em;
  text-transform: uppercase; color: #8b9d99; text-align: right; padding: 0 .6rem .5rem;
  border-bottom: 1px solid #c6d2cf; vertical-align: bottom; line-height: 1.35; }
table.vh th:first-child, table.vh td:first-child { text-align: left; padding-left: 0; }
table.vh td { padding: .5rem .6rem; border-bottom: 1px solid #dbe3e1; text-align: right; }
table.vh tbody tr:last-child td { border-bottom: 0; }
table.vh td.nm { font-family: "IBM Plex Sans", sans-serif; font-weight: 500; white-space: nowrap; }
table.vh tr.good td { background: #d3e7e3; }
table.vh tr.bad td { background: #f3ddd9; }
.vh-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .5rem; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dbe3e1; }
section[data-testid="stSidebar"] h2 { font-size: 1.05rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: .1rem; border-bottom: 1px solid #dbe3e1; }
.stTabs [data-baseweb="tab"] { font-family: "IBM Plex Mono", monospace; font-size: .8rem;
  padding: .55rem .9rem; color: #61756f; }
.stTabs [aria-selected="true"] { color: #0d6f66 !important; font-weight: 500; }
div[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; }
hr { border-color: #dbe3e1; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# computation (cached; every widget below reads from these)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_suites(n_ep, n_turns, corrupt_p=0.0):
    out = {}
    for name, cls in AGENTS:
        out[name] = [run_episode(cls(), PERSONAS[i % len(PERSONAS)], seed=i,
                                 n_turns=n_turns, corrupt_p=corrupt_p)
                     for i in range(n_ep)]
    return out


@st.cache_data(show_spinner=False)
def panel_states(n_ep, n_turns):
    ref = [run_episode(OracleAgent(), PERSONAS[i % len(PERSONAS)], seed=500 + i,
                       n_turns=n_turns) for i in range(n_ep)]
    return [(t.index, t.user_before) for ep in ref for t in ep.turns][::3]


@st.cache_data(show_spinner=False)
def headline(n_ep, n_turns):
    suites = build_suites(n_ep, n_turns)
    states = panel_states(n_ep, n_turns)
    rows, pp, oc = [], [], []
    for name, cls in AGENTS:
        p, c = M.turn_panel(cls, states)
        tl = M.tail_load(suites[name])
        m, reg = M.mimicry_and_regulation(suites[name])
        slope, _ = M.calibration_drift(suites[name])
        pp.append(p)
        oc.append(-tl)
        rows.append(dict(agent=name, panel=p, panel_cal=c, tail=tl,
                         fail=M.failure_rate(suites[name]), mimicry=m,
                         regulation=reg, drift=slope))
    return rows, M.spearman(pp, oc), len(states)


@st.cache_data(show_spinner=False)
def one_episode(agent_name, persona_name, n_turns, seed=3):
    cls = dict(AGENTS)[agent_name]
    persona = [p for p in PERSONAS if p.name == persona_name][0]
    return run_episode(cls(), persona, seed=seed, n_turns=n_turns)


@st.cache_data(show_spinner=False)
def uptake_curves(n_ep, n_turns):
    suites = build_suites(n_ep, n_turns)
    lags = list(range(1, 26))
    return {n: M.feedback_uptake(suites[n], lags=lags) for n, _ in AGENTS}, lags


@st.cache_data(show_spinner=False)
def corrupted_rows(n_ep, n_turns):
    clean = build_suites(n_ep, n_turns)
    c20 = build_suites(n_ep, n_turns, 0.2)
    c40 = build_suites(n_ep, n_turns, 0.4)
    return [dict(agent=n, clean=M.feedback_uptake(clean[n])[10],
                 c20=M.feedback_uptake(c20[n])[10], c40=M.feedback_uptake(c40[n])[10])
            for n, _ in AGENTS if n != "oracle"]


@st.cache_data(show_spinner=False)
def localization(n_ep, n_turns, severity):
    import random
    rng = random.Random(11)
    cases = []
    for i in range(n_ep):
        persona = PERSONAS[i % len(PERSONAS)]
        ft = rng.randrange(6, max(8, n_turns - 8))
        cases.append((run_episode(CalibratedAgent(), persona, seed=2000 + i,
                                  n_turns=n_turns, fault_turn=ft,
                                  fault_severity=severity), persona))
    res = score_localization(cases, tolerance=1)
    ep, persona = cases[0]
    onset, ranked = localize(ep, persona)
    return res, dict(persona=ep.persona, true=ep.true_fault_turn, predicted=onset,
                     ranked=ranked[:5], scores=anomaly_scores(ep))


@st.cache_data(show_spinner=False)
def fp_rate(n_ep, n_turns):
    suites = build_suites(n_ep, n_turns)
    return false_positive_rate([(ep, PERSONAS[i % len(PERSONAS)])
                                for i, ep in enumerate(suites["calibrated"])])


@st.cache_data(show_spinner=False)
def sigma_between(n_ep, n_turns):
    suites = build_suites(n_ep, n_turns)
    return M.stdev([1.0 + 6.0 * e.mean_perceived for e in suites["calibrated"]])


def base_fig(height=340, xtitle="", ytitle=""):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=12, color=MUTED),
        hoverlabel=dict(font_family="IBM Plex Mono, monospace", font_size=12,
                        bgcolor=SURFACE, bordercolor=RULE, font_color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=xtitle, gridcolor=RULE, zerolinecolor=RULE,
                   linecolor=RULE, tickfont=dict(color=FAINT)),
        yaxis=dict(title=ytitle, gridcolor=RULE, zerolinecolor=RULE,
                   linecolor=RULE, tickfont=dict(color=FAINT)),
    )
    return fig


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Suite parameters")
    st.caption("Nothing here is precomputed. Move these and every table and "
               "chart on the page is recalculated from a fresh run.")
    n_ep = st.slider("Conversations per policy", 10, 80, 30, 5)
    n_turns = st.slider("Turns per conversation", 20, 80, 40, 5)
    st.markdown("---")
    st.markdown(
        '<div class="vh-note">Five simulated callers &times; {} conversations '
        '&times; {} turns = <b>{:,} scored turns</b> per policy, five policies. '
        'Deterministic: the same settings always give the same numbers.</div>'
        .format(n_ep, n_turns, n_ep * n_turns), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="vh-note"><b>Vahit Feryad, PhD</b><br>'
        '<a href="https://github.com/vahit19/voicehaul">Source</a> &middot; '
        '<a href="https://github.com/vahit19/runopsy">Runopsy</a> &middot; '
        '<a href="https://github.com/vahit19/LongHaul-Bench">LongHaul-Bench</a><br>'
        '<a href="https://scholar.google.com/citations?hl=en&user=JUtYZ1oAAAAJ">'
        'Google Scholar</a> &middot; '
        '<a href="https://www.linkedin.com/in/vahit-feryad-19517256/">LinkedIn</a>'
        '</div>', unsafe_allow_html=True)

rows, rho, n_states = headline(n_ep, n_turns)
by_agent = {r["agent"]: r for r in rows}

# ---------------------------------------------------------------------------
# masthead
# ---------------------------------------------------------------------------

st.markdown('<div class="vh-kicker">VoiceHaul &middot; long-horizon evaluation '
            'for empathic voice agents</div>', unsafe_allow_html=True)
st.markdown('<h1 style="margin:.2rem 0 .8rem">Sounding right on every turn,<br>and getting the conversation wrong.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="vh-lede">A turn-level rating tells you whether a response sounded '
    'right. It cannot tell you whether a model still honours what the caller asked '
    'for twenty turns ago, whether its calibration decays as a session runs long, '
    'whether it is regulating the caller\'s affect or mirroring it back, or which '
    'turn broke a call that ended badly.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="vh-note" style="max-width:60ch">This measures those four things, '
    'then measures how much rating budget you need before any of them is '
    'detectable. Everything below is running live in this browser tab.</p>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="vh-panel" style="max-width:68ch;background:#f6f8f7">'
    '<div class="vh-kicker" style="margin-bottom:.4rem">Provenance</div>'
    '<div class="vh-note">The measurement design here is not new work. It is the '
    'method from <a href="https://github.com/vahit19/LongHaul-Bench">LongHaul-Bench</a> '
    '(a 1,000+ episode long-horizon reliability benchmark, five-world experimental '
    'programme, memory-strategy ablations) and '
    '<a href="https://github.com/vahit19/runopsy">Runopsy</a> (causal failure-onset '
    'diagnosis with counterfactual replay), carried across from text agents to '
    'voice. Building this instance took a day because the underlying method took '
    'two years. <b>What you are looking at is phase zero</b> — the synthetic '
    'validation step that has to pass before any of it is pointed at real audio. '
    'The <i>Where this goes</i> tab is the part that matters.</div></div>',
    unsafe_allow_html=True)

hard = localization(n_ep, n_turns, 0.35)[0]
st.markdown(
    '<div class="vh-cards">'
    '<div><div class="k">turn panel vs outcome</div>'
    '<div class="v" style="color:{alarm}">&rho; = {rho:+.2f}</div>'
    '<div class="s">the two rankings are anti-correlated</div></div>'
    '<div><div class="k">onset localization</div>'
    '<div class="v" style="color:{accent}">{top1:.0%}</div>'
    '<div class="s">top-1 at the hardest fault severity, {fp:.0%} false positives</div></div>'
    '<div><div class="k">this run</div><div class="v">{turns:,} turns</div>'
    '<div class="s">5 policies &times; 5 callers, recomputed on load</div></div>'
    '</div>'.format(alarm=ALARM, accent=ACCENT, rho=rho, top1=hard["top1"],
                    fp=fp_rate(n_ep, n_turns), turns=n_ep * n_turns * 5),
    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# the finding
# ---------------------------------------------------------------------------

st.markdown("## The same five models, ranked two ways")
st.markdown(
    '<div class="vh-prose"><p>On the left, a fixed-context turn panel: every '
    'policy answers the same {} held-out caller states and each turn is rated on '
    'its own, the way a prompt-set leaderboard works. On the right, the same '
    'policies actually holding {}-turn conversations.</p></div>'
    .format(n_states, n_turns), unsafe_allow_html=True)

best_panel = max(rows, key=lambda r: r["panel"])
body = ""
for r in rows:
    cls = "bad" if r["agent"] == best_panel["agent"] else ("good" if r["fail"] == 0 else "")
    body += ('<tr class="{cls}"><td class="nm"><span class="vh-dot" '
             'style="background:{c}"></span>{n}</td>'
             '<td>{p:.3f}</td><td>{pc:.3f}</td><td>{tl:.3f}</td><td>{f:.0%}</td>'
             '<td>{d:+.2f}</td></tr>').format(
        cls=cls, c=AGENT_COLOR[r["agent"]], n=AGENT_LABEL[r["agent"]],
        p=r["panel"], pc=r["panel_cal"], tl=r["tail"], f=r["fail"], d=r["drift"])
st.markdown(
    '<div class="vh-panel"><table class="vh"><thead><tr><th>policy</th>'
    '<th>panel:<br>perceived empathy</th><th>panel:<br>calibration</th>'
    '<th>conversation:<br>left-over distress</th>'
    '<th>conversation:<br>fail rate</th>'
    '<th>calibration drift<br>pts / 10 turns</th></tr></thead>'
    '<tbody>' + body + '</tbody></table></div>', unsafe_allow_html=True)

best_conv = min(rows, key=lambda r: r["tail"])
st.markdown(
    '<div class="vh-callout"><p style="margin:0">The turn panel says the best '
    'model is <b>{bp}</b>, which fails {bpf:.0%} of conversations. The '
    'conversations say it is <b>{bc}</b>. Rank correlation between the two '
    'orderings: <b>&rho; = {rho:+.2f}</b>.</p></div>'
    '<div class="vh-prose"><p>Not because turn-level rating is wrong — it is the '
    'right instrument for the question it asks. It is because a turn panel holds '
    'the context fixed, and in a real conversation <b>the model creates the '
    'context it is later scored on</b>. A model that keeps callers agitated is '
    'subsequently asked easier-looking questions.</p></div>'.format(
        bp=AGENT_LABEL[best_panel["agent"]], bpf=best_panel["fail"],
        bc=AGENT_LABEL[best_conv["agent"]], rho=rho), unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Watch a conversation", "Does a request survive?", "Mimicry vs regulation",
    "Which turn broke it", "Rating budget", "What's real here",
    "Where this goes"])

# --- tab 1 -----------------------------------------------------------------
with t1:
    st.markdown("### Watch one conversation happen")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">Pick a policy and a caller. The '
        'trace shows what the caller is left carrying, what a turn-level rater '
        'would score, and what the turn was actually worth. Try <b>mirror</b> '
        'against the <b>hostile caller</b> first.</p></div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        agent_name = st.radio("Policy under test", [a for a, _ in AGENTS],
                              format_func=lambda a: AGENT_LABEL[a],
                              horizontal=True, index=0, key="ep_agent")
    with c2:
        persona_name = st.radio("Caller", [p.name for p in PERSONAS],
                                format_func=lambda p: PERSONA_LABEL[p],
                                horizontal=True, index=1, key="ep_persona")
    st.caption(AGENT_BLURB[agent_name])

    ep = one_episode(agent_name, persona_name, n_turns)
    xs = list(range(len(ep.turns)))
    fig = base_fig(360, "turn", "level")
    fig.update_yaxes(range=[-0.02, 1.05])
    for key, label, color, width in [
            ("d", "distress the caller is left carrying", ALARM, 3),
            ("p", "perceived empathy — what a turn-level rater scores", AMBER, 1.8),
            ("c", "calibration — what the turn was actually worth", ACCENT, 1.8)]:
        vals = [{"d": t.user_after.negative_load, "p": t.perceived,
                 "c": t.calibration}[key] for t in ep.turns]
        fig.add_trace(go.Scatter(x=xs, y=vals, name=label, mode="lines",
                                 line=dict(color=color, width=width),
                                 hovertemplate="turn %{x}<br>%{y:.2f}<extra></extra>"))
    for t in ep.turns:
        if t.new_directive:
            fig.add_vline(x=t.index, line=dict(color=ACCENT, width=1, dash="dot"),
                          opacity=0.45)
    st.plotly_chart(fig, use_container_width=True)

    if ep.failed:
        st.markdown(
            '<div class="vh-panel" style="border-color:{a};background:{s}">'
            '<b style="color:{a}">Conversation failed.</b> The caller is left '
            'carrying {v:.2f} of negative affect after {n} turns. Dotted lines '
            'mark every turn where the caller made an explicit request.</div>'
            .format(a=ALARM, s=ALARM_SOFT, v=ep.turns[-1].user_after.negative_load,
                    n=len(ep.turns)), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="vh-panel" style="border-color:{a};background:{s}">'
            '<b style="color:{a}">Conversation held.</b> The caller is left '
            'carrying {v:.2f} of negative affect after {n} turns. Dotted lines '
            'mark every turn where the caller made an explicit request.</div>'
            .format(a=ACCENT, s=ACCENT_SOFT, v=ep.turns[-1].user_after.negative_load,
                    n=len(ep.turns)), unsafe_allow_html=True)

    with st.expander("Turn-by-turn log"):
        st.dataframe(
            [{"turn": t.index,
              "distress left": round(t.user_after.negative_load, 2),
              "perceived": round(t.perceived, 2),
              "calibration": round(t.calibration, 2),
              "speech rate": round(t.action.speech_rate, 2),
              "cheerfulness": round(t.action.cheerfulness, 2),
              "acknowledgement": round(t.action.acknowledgement, 2),
              "what the caller did": (
                  ("new grievance; " if t.shock > 0 else "") +
                  ('asked: "' + t.new_directive.replace("_", " ") + '"'
                   if t.new_directive else "")).strip("; ")}
             for t in ep.turns],
            width="stretch", hide_index=True, height=340)

    st.markdown(
        '<div class="vh-prose"><h3>What to look for</h3>'
        '<p>With <b>mirror</b> against the <b>hostile caller</b>, perceived empathy '
        'stays respectable the whole way down — it sounds attuned on every single '
        'turn — while distress never comes down. It matches the caller\'s energy '
        'instead of sitting just below it, so the caller has nowhere to come down '
        'to. No turn is bad. The conversation is.</p>'
        '<p>With <b>drifter</b>, watch what happens a dozen turns after a request: '
        'it complies for a while, then quietly stops, and speech rate creeps back '
        'up. The turn-level score barely moves.</p></div>', unsafe_allow_html=True)

# --- tab 2 -----------------------------------------------------------------
with t2:
    st.markdown("### Does a request survive the conversation?")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">Callers voice explicit requests — '
        '<i>slow down</i>, <i>stop apologising</i>, <i>be concise</i>. Uptake is '
        'the share still honoured some number of turns later. Two policies with '
        'the same turn-level score can sit at opposite ends of this curve.</p></div>',
        unsafe_allow_html=True)

    curves, lags = uptake_curves(n_ep, n_turns)
    fig = base_fig(380, "turns since the request", "share of requests still honoured")
    fig.update_yaxes(range=[-0.03, 1.05], tickformat=".0%")
    for name, _ in AGENTS:
        ys = [curves[name][l] if curves[name][l] == curves[name][l] else None
              for l in lags]
        fig.add_trace(go.Scatter(x=lags, y=ys, name=AGENT_LABEL[name], mode="lines",
                                 line=dict(color=AGENT_COLOR[name], width=2.4),
                                 hovertemplate="lag %{x}<br>%{y:.0%}<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)

    d = curves["drifter"]
    st.markdown(
        '<div class="vh-callout"><p style="margin:0">The drifting policy honours '
        '<b>{a:.0%}</b> of requests immediately and <b>{b:.0%}</b> twenty turns '
        'later, at an essentially unchanged turn-level score. Only the lag exposes '
        'it — which is why this cannot be measured one turn at a time.</p></div>'
        .format(a=d[1], b=d[20]), unsafe_allow_html=True)

# --- tab 3 -----------------------------------------------------------------
with t3:
    st.markdown("### Mimicry is not regulation")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">Two readings of the same '
        'behaviour. Horizontally: how closely the policy\'s vocal energy tracks '
        'the caller\'s — how attuned it <i>sounds</i>. Vertically: how much '
        'negative affect it actually removes per turn. A single "empathy" score '
        'collapses these into one number and loses the distinction that '
        'matters.</p></div>', unsafe_allow_html=True)

    fig = base_fig(420, "mimicry — correlation with the caller's vocal energy",
                   "regulation — negative affect removed per turn")
    fig.add_hline(y=0, line=dict(color=RULE, width=1))
    fig.add_vline(x=0, line=dict(color=RULE, width=1))
    for r in rows:
        fig.add_trace(go.Scatter(
            x=[r["mimicry"]], y=[r["regulation"]], mode="markers+text",
            name=AGENT_LABEL[r["agent"]], text=[AGENT_LABEL[r["agent"]]],
            textposition="middle right", textfont=dict(color=INK2, size=12),
            marker=dict(size=15, color=AGENT_COLOR[r["agent"]]), showlegend=False,
            hovertemplate=(AGENT_LABEL[r["agent"]] + "<br>mimicry %{x:+.2f}"
                           "<br>regulation %{y:.2f}<extra></extra>")))
    fig.update_xaxes(range=[-1.05, 0.75])
    st.plotly_chart(fig, use_container_width=True)

    mi, ca = by_agent["mirror"], by_agent["calibrated"]
    st.markdown(
        '<div class="vh-callout"><p style="margin:0">The mirror tracks the caller '
        '(<b>{m:+.2f}</b>). The calibrated policy moves against them '
        '(<b>{c:+.2f}</b>) and removes <b>{p:.0f}%</b> more distress per turn. '
        'Both would be described as empathic by a listener.</p></div>'.format(
            m=mi["mimicry"], c=ca["mimicry"],
            p=100 * (ca["regulation"] / mi["regulation"] - 1) if mi["regulation"] else 0),
        unsafe_allow_html=True)

    st.markdown("#### When the feedback channel itself is wrong")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">A share of caller requests reach '
        'the model as the <i>opposite</i> instruction — an ASR error, or an '
        'adversarial caller. The caller still expects the original.</p></div>',
        unsafe_allow_html=True)
    cr = corrupted_rows(n_ep, n_turns)
    body = ""
    for r in cr:
        body += ('<tr><td class="nm"><span class="vh-dot" style="background:{c}">'
                 '</span>{n}</td><td>{a:.2f}</td><td>{b:.2f}</td><td>{d:.2f}</td>'
                 '<td>{dr:+.1f} pts</td></tr>').format(
            c=AGENT_COLOR[r["agent"]], n=AGENT_LABEL[r["agent"]], a=r["clean"],
            b=r["c20"], d=r["c40"], dr=100 * (r["c40"] - r["clean"]))
    st.markdown('<div class="vh-panel"><table class="vh"><thead><tr>'
                '<th>policy</th><th>clean channel</th><th>20% corrupted</th>'
                '<th>40% corrupted</th><th>drop</th></tr></thead><tbody>'
                + body + '</tbody></table></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">The policies that listen best '
        'degrade most. Perfect compliance with a corrupted channel is itself a '
        'failure mode, and it is invisible to any suite whose feedback channel is '
        'assumed clean.</p></div>', unsafe_allow_html=True)

# --- tab 4 -----------------------------------------------------------------
with t4:
    st.markdown("### Which turn broke it")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">A regression is injected at a '
        'turn the diagnostic is never told about: the model loses its conditioning '
        'and reverts to a generic upbeat persona. Four cheap deterministic signals '
        'propose candidates, a walk-back finds where the anomalous stretch begins, '
        'and counterfactual replay <i>gates</i> — if repairing the model from the '
        'proposed turn would not have changed the outcome, it returns no answer '
        'rather than a wrong one.</p></div>', unsafe_allow_html=True)

    sev = st.select_slider(
        "Fault severity — 1.00 is a full reversion, 0.35 blends the fault into the "
        "model's own policy and is the realistic case",
        options=[0.35, 0.50, 0.70, 1.00], value=1.00)
    res, ex = localization(n_ep, n_turns, sev)

    fig = base_fig(300, "turn", "anomaly score")
    colors = [ALARM if i == ex["true"] else "#c6d2cf"
              for i in range(len(ex["scores"]))]
    fig.add_trace(go.Bar(x=list(range(len(ex["scores"]))), y=ex["scores"],
                         marker_color=colors, showlegend=False,
                         hovertemplate="turn %{x}<br>score %{y:.2f}<extra></extra>"))
    fig.add_annotation(x=ex["true"], y=max(ex["scores"]),
                       text="fault injected at turn {} · predicted {}".format(
                           ex["true"], ex["predicted"]),
                       showarrow=True, arrowhead=0, arrowcolor=ALARM, ay=-30,
                       font=dict(color=ALARM, size=12))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("top-1 accuracy (±1 turn)", "{:.1%}".format(res["top1"]))
    c2.metric("top-3 accuracy", "{:.1%}".format(res["top3"]))
    c3.metric("median error", "{:+.0f} turns".format(res["median_signed_error"]))
    c4.metric("false positives on healthy runs",
              "{:.1%}".format(fp_rate(n_ep, n_turns)))

    st.markdown(
        '<div class="vh-prose"><p class="vh-note">Accuracy should always be '
        'reported against severity, never as a single number. Move the slider to '
        '0.35 and watch it fall — that honesty is the difference between a '
        'diagnostic you can deploy and a demo number.</p></div>',
        unsafe_allow_html=True)

# --- tab 5 -----------------------------------------------------------------
with t5:
    st.markdown("### How many conversations before a regression is visible?")
    st.markdown(
        '<div class="vh-prose"><p class="vh-note">Human ratings are both the '
        'ground truth and the budget line. Between-conversation spread is measured '
        'from this suite; per-rater noise is yours to set. This is the calculation '
        'that turns "we track regressions" into a number of conversations and a '
        'cost.</p></div>', unsafe_allow_html=True)

    sb = sigma_between(n_ep, n_turns)
    c1, c2, c3 = st.columns(3)
    with c1:
        n_conv = st.slider("Conversations per arm", 10, 600, 100, 10)
    with c2:
        n_raters = st.slider("Raters per conversation", 1, 12, 3, 1)
    with c3:
        rater_sd = st.slider("Rater noise (Likert sd)", 0.4, 1.6, 0.9, 0.05)

    mde = M.min_detectable_effect(sb, rater_sd, n_conv, n_raters)
    c1, c2, c3 = st.columns(3)
    c1.metric("Smallest detectable regression", "{:.2f}".format(mde),
              help="Likert points, two-sample, alpha = 0.05, power = 0.80")
    c2.metric("Conversations for a 0.20-point regression",
              "{:,}".format(M.required_n(0.20, sb, rater_sd, n_raters)))
    c3.metric("Measured between-conversation sd", "{:.2f}".format(sb))

    Ns = [30, 100, 300, 1000]
    head = "".join("<th>N = {}</th>".format(n) for n in Ns)
    body = ""
    for rr in (1, 3, 5, 10):
        cls = "good" if rr == n_raters else ""
        body += '<tr class="{}"><td class="nm">{}</td>'.format(cls, rr) + "".join(
            "<td>{:.2f}</td>".format(M.min_detectable_effect(sb, rater_sd, n, rr))
            for n in Ns) + "</tr>"
    st.markdown('<div class="vh-panel"><table class="vh"><thead><tr>'
                '<th>raters / conversation</th>' + head + '</tr></thead><tbody>'
                + body + '</tbody></table>'
                '<p class="vh-note" style="margin:.7rem 0 0">Cells are the '
                'smallest true regression, in Likert points, the suite can detect '
                'at 80% power.</p></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="vh-callout"><p style="margin:0">At your current settings a '
        '{n}-conversation suite cannot see anything smaller than <b>{m:.2f}</b> '
        'points. Every regression below that line ships.</p></div>'.format(
            n=n_conv, m=mde), unsafe_allow_html=True)

# --- tab 6 -----------------------------------------------------------------
with t6:
    st.markdown("### What is real and what is simulated")
    st.markdown(
        '<div class="vh-prose">'
        '<p><b>Simulated:</b> the callers, the policies, and the affect dynamics. '
        'The five policies are deterministic simulators, not language models — '
        'each embodies exactly one known failure mode. The caller model and both '
        'scoring functions are written by hand.</p>'
        '<p><b>Real:</b> the metrics, the estimators, the statistics, and the '
        'localization algorithm. Those are the deliverable.</p>'
        '<p>The reason for a synthetic environment is not convenience. <b>You '
        'cannot validate a measurement instrument without ground truth you '
        'control.</b> If you only ever run an eval against real models, a metric '
        'that reports the wrong thing and a model that behaves badly are '
        'indistinguishable. Here the fault turn is known, the failure mode is '
        'known, and the ideal policy is known — so "93% accurate at severity 0.5 '
        'with no false positives" is a checkable statement about the <i>method</i>, '
        'not a leaderboard entry.</p></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="vh-callout warn"><p style="margin:0">The rank-correlation '
        'result is the one to read sceptically. I encoded the hypothesis that '
        'turn-level raters reward attunement and warmth while outcomes reward '
        'down-regulation, and the environment then confirms it — which is close to '
        'circular on its own. Its value is that the hypothesis is now '
        '<i>falsifiable</i> against real rater data: fit the perceived-empathy '
        'function to real human ratings, keep the outcome measure, and the same '
        'code reports whether the gap survives. If it does not, that is a genuinely '
        'useful negative result about a class of eval suites.</p></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="vh-prose"><h3>Plugging in a real model</h3>'
        '<p>The design property that makes this practical: the harness needs no '
        'privileged access to the model under test. Both sides of the conversation '
        'are scored from audio — the caller\'s affect trajectory from expression '
        'measurement on the caller channel, and the model\'s delivery parameters '
        '(speech rate, prosodic positivity, apology rate, verbosity, '
        'acknowledgement) from expression measurement plus transcript statistics on '
        'the model channel. The same metrics apply to a model you own, one you '
        'licence, and a competitor\'s public endpoint.</p>'
        '<p>Two things I would fix before trusting it on real audio, in order. The '
        'compact affect basis is currently a hand-written alias map over a '
        '48-category readout; that projection should be <i>fitted</i> against human '
        'ratings of the same clips, because everything downstream inherits its '
        'error. And apology and acknowledgement detection uses English keyword '
        'lists; for a multilingual suite that has to become a small classifier. The '
        'prosodic features generalise, the lexical ones do not.</p>'
        '<h3>Running it yourself</h3>'
        '<p>The harness is pure Python standard library — no dependencies, no API '
        'key, no network. <code>python run_demo.py</code> produces this whole '
        'report as text in about two seconds, and '
        '<code>python test_voicehaul.py</code> runs sixteen property checks on the '
        'harness itself, because an evaluation harness that is wrong is worse than '
        'no harness: it is believed.</p></div>', unsafe_allow_html=True)

# --- tab 7 -----------------------------------------------------------------
with t7:
    st.markdown("### What this is phase zero of")
    st.markdown(
        '<div class="vh-prose">'
        '<p>Everything on the other tabs is the part that can be done in a '
        'synthetic environment, and it is the smallest part. It establishes that '
        'the instrument measures what it claims to measure on a case where the '
        'answer is known by construction. That is a precondition, not a '
        'result.</p>'
        '<p>The four questions below are the actual research. Each is a quarter or '
        'more of work, each is gated on data that does not exist yet, and each '
        'produces both a publishable finding and a product surface. I am listing '
        'them with their cost rather than their promise, because a benchmark that '
        'is oversold once is never trusted again.</p>'
        '</div>', unsafe_allow_html=True)

    QUESTIONS = [
        dict(
            n="01",
            q="Does the gap survive contact with real raters?",
            why="The headline result rests on a hypothesis I encoded by hand: that "
                "turn-level raters reward attunement and warmth while conversation "
                "outcomes reward down-regulation. In a synthetic world that is "
                "close to circular. Against real ratings it becomes falsifiable, "
                "and it is the one claim everything else depends on.",
            needs="Several hundred real conversations, each rated twice: turn by "
                  "turn in isolation, and once at the conversation level. Multiple "
                  "raters per item, because the power table on the previous tab is "
                  "what says how many. This is a rating-panel study, not a code "
                  "change.",
            makes="Either a validated instrument, or a clean negative result about "
                  "a whole class of evaluation suites. Both are worth publishing; "
                  "one of them is worth building a product on.",
            time="One to two quarters, almost all of it data collection and "
                 "inter-rater reliability work"),
        dict(
            n="02",
            q="Is the measurement invariant across languages?",
            why="Down-regulation is a cultural norm as much as an acoustic one. "
                "Speaking below someone's energy may not soothe identically in "
                "every language, and the lexical features here are English-only by "
                "construction. If the instrument is not invariant, a "
                "cross-language leaderboard compares different quantities and "
                "calls it one score. For a fifty-language product that is not a "
                "footnote.",
            needs="Parallel evaluation corpora across several typologically "
                  "distinct languages, native-speaker rater panels, and formal "
                  "measurement-invariance testing - configural, metric, scalar.",
            makes="A methods paper of the kind cited by everyone who later builds "
                  "a multilingual voice benchmark, and a defensible answer to the "
                  "first question a serious enterprise buyer asks.",
            time="Two to three quarters, running partly in parallel with 01"),
        dict(
            n="03",
            q="Can the affect projection be learned instead of hand-written?",
            why="A 48-category expression readout is collapsed onto a "
                "six-dimension basis by an alias map I wrote by hand. Every metric "
                "downstream inherits that projection's error and nobody currently "
                "knows how large it is. This is the least glamorous item here and "
                "probably the highest-leverage one.",
            needs="Paired data - expression measurement and human ratings on the "
                  "same clips - then fitting the projection against the ratings "
                  "rather than asserting it, with held-out validation.",
            makes="A measurable accuracy improvement across every metric at once, "
                  "and a reusable component rather than a bespoke one.",
            time="A quarter once the data from 01 exists; it cannot start before"),
        dict(
            n="04",
            q="Does optimising for these metrics actually produce better models?",
            why="The real test of an evaluation is not whether it correlates with "
                "human judgement. It is whether using it as a training signal "
                "moves a model on held-out human judgement. This is where "
                "evaluation stops being measurement and becomes post-training, and "
                "it is the only one of these four that changes what a model is "
                "rather than what we know about it.",
            needs="Everything above, plus a model that can actually be fine-tuned "
                  "and a held-out human evaluation that was never used for "
                  "training.",
            makes="The finding that would justify the whole programme, or the "
                  "finding that these metrics are diagnostic but not optimisable - "
                  "which is itself important and rarely reported.",
            time="Four quarters or more, and it should not be started early"),
    ]

    for item in QUESTIONS:
        st.markdown(
            '<div class="vh-panel">'
            '<div style="display:flex;gap:1.1rem;align-items:baseline">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:1.45rem;'
            'color:{c};font-weight:500;line-height:1">{n}</div>'
            '<div style="flex:1">'
            '<div style="font-family:IBM Plex Serif,Georgia,serif;font-size:1.12rem;'
            'font-weight:600;margin-bottom:.55rem;line-height:1.3">{q}</div>'
            '<p class="vh-note" style="margin:.2rem 0 .6rem">'
            '<b>Why it matters.</b> {why}</p>'
            '<p class="vh-note" style="margin:0 0 .6rem">'
            '<b>What answering it takes.</b> {needs}</p>'
            '<p class="vh-note" style="margin:0 0 .7rem">'
            '<b>What it produces.</b> {makes}</p>'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
            'letter-spacing:.09em;text-transform:uppercase;color:{c}">{time}</div>'
            '</div></div></div>'.format(c=ACCENT, **item), unsafe_allow_html=True)

    st.markdown(
        '<div class="vh-callout"><p style="margin:0">None of these four compresses '
        'by working harder: three are gated on rating data that has to be '
        'collected, and the fourth is gated on the other three. That is the shape '
        'of the work - not a tool to be delivered, but an instrument that earns '
        'authority by being maintained, versioned, contamination-checked and '
        're-validated as the models it measures keep moving. A benchmark nobody '
        'maintains stops being cited in about a year.</p></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="vh-prose"><h3>Why the fast part was fast</h3>'
        '<p>The measurement design here is not new. It is the method from two '
        'projects that took about two years between them. '
        '<a href="https://github.com/vahit19/LongHaul-Bench">LongHaul-Bench</a> ran '
        'a five-world experimental programme over a thousand-plus sequential '
        'episodes per run under hard edge constraints, comparing frozen, '
        'append-only, reflect, gated and oracle memory strategies with full '
        'ablations and statistical analysis. '
        '<a href="https://github.com/vahit19/runopsy">Runopsy</a> is the causal '
        'failure-onset method that the fourth tab is a port of.</p>'
        '<p>That is why this instance came together quickly, and it is also why I '
        'would not describe it as quick work. The fast part is the part that was '
        'already solved. What is left - validating a measurement instrument '
        'against human judgement, establishing that it holds across languages, and '
        'finding out whether it can be optimised against - is ordinary research on '
        'an ordinary research timescale. That is the part I am interested in '
        'doing.</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div class="vh-note">Built by <b>Vahit Feryad, PhD</b> — applied AI research '
    'engineer, evaluation and reliability. Apache-2.0. '
    '<a href="https://github.com/vahit19/voicehaul">github.com/vahit19/voicehaul</a>'
    '</div>', unsafe_allow_html=True)
