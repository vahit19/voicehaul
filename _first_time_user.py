"""A first-time reader opens this page knowing nothing. Fix what that reader
cannot answer for themselves: what the five policies are, what the callers are,
what a number means, and what happens when they press a button."""
import io

p = "app/app.py"
s = io.open(p, encoding="utf-8").read()

# --------------------------------------------------------- legends ---------
LEGEND = '''
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
        '<div style="display:flex;gap:.7rem;align-items:flex-start;'
        'padding:.45rem 0;border-bottom:1px solid #eef2f1">'
        '<div style="width:9px;height:9px;border-radius:50%;background:{c};'
        'margin-top:.35rem;flex:0 0 auto"></div>'
        '<div><b style="font-family:IBM Plex Mono,monospace;font-size:.84rem">'
        '{k}</b> <span style="color:{c};font-size:.8rem">{t}</span>'
        '<div style="font-size:.85rem;color:#61756f;line-height:1.45">{d}</div>'
        '</div></div>').format(c=c, k=k, t=t, d=d)
        for k, (t, c, d) in entries.items())
    return ('<details style="margin:.2rem 0 .8rem;font-family:IBM Plex Sans,'
            'system-ui,sans-serif"><summary style="cursor:pointer;font-size:.86rem;'
            'color:#0d6f66">{}</summary><div style="margin-top:.5rem">{}</div>'
            '</details>').format(title, rows)


def policy_legend():
    return _legend(POLICY_BLURB, "What are these five policies?")


def caller_legend():
    return _legend({k: ("", MUTED, v) for k, v in CALLER_BLURB.items()},
                   "Who are the five callers?")


'''
s = s.replace("VERDICT_STYLE = {", LEGEND + "VERDICT_STYLE = {")

# --------------------------------------------- orientation under the title --
s = s.replace('''    gr.HTML(headline_strip())''',
'''    gr.Markdown(
        "*Six tabs, each one already showing a result &mdash; the controls are "
        "there to change the question, not to make something appear. Nothing "
        "here calls an external service; the harness is pure Python and runs "
        "in this tab.*")
    gr.HTML(headline_strip())''')

# ----------------------------------------------------- legends in the tabs --
s = s.replace('''            with gr.Row():
                g_base = gr.Dropdown(POLICIES, value="calibrated", label="baseline")
                g_cand = gr.Dropdown(POLICIES, value="mirror", label="candidate")
                g_ep = gr.Slider(10, 60, 30, step=5, label="conversations")
                g_tn = gr.Slider(20, 60, 40, step=5, label="turns")''',
'''            gr.HTML(policy_legend())
            with gr.Row():
                g_base = gr.Dropdown(POLICIES, value="calibrated",
                                     label="baseline (what is in production)")
                g_cand = gr.Dropdown(POLICIES, value="mirror",
                                     label="candidate (what you want to ship)")
                g_ep = gr.Slider(10, 60, 30, step=5,
                                 label="conversations per arm")
                g_tn = gr.Slider(20, 60, 40, step=5, label="turns per conversation")''')

s = s.replace('''            with gr.Row():
                c_pol = gr.Dropdown(POLICIES, value="mirror", label="policy")
                c_per = gr.Dropdown(list(PLABEL.values()), value="hostile caller",
                                    label="caller")
                c_tn = gr.Slider(20, 60, 40, step=5, label="turns")''',
'''            gr.HTML(policy_legend())
            gr.HTML(caller_legend())
            with gr.Row():
                c_pol = gr.Dropdown(POLICIES, value="mirror", label="policy")
                c_per = gr.Dropdown(list(PLABEL.values()), value="hostile caller",
                                    label="caller")
                c_tn = gr.Slider(20, 60, 40, step=5, label="turns")''')

s = s.replace('''            with gr.Row():
                o_sev = gr.Slider(0.35, 1.0, 1.0, step=0.05, label="fault severity")
                o_ft = gr.Slider(6, 30, 18, step=1, label="inject at turn")
                o_tn = gr.Slider(24, 60, 40, step=4, label="turns")''',
'''            with gr.Row():
                o_sev = gr.Slider(0.35, 1.0, 1.0, step=0.05,
                                  label="fault severity - 1.00 obvious, 0.35 subtle")
                o_ft = gr.Slider(6, 30, 18, step=1,
                                 label="inject the fault at this turn")
                o_tn = gr.Slider(24, 60, 40, step=4, label="turns per conversation")''')

s = s.replace('''                b_n = gr.Slider(20, 600, 100, step=20, label="conversations per arm")
                b_r = gr.Slider(1, 10, 3, step=1, label="raters per conversation")
                b_sd = gr.Slider(0.4, 1.6, 0.9, step=0.05, label="rater noise (Likert sd)")
                b_ep = gr.Slider(20, 60, 40, step=10, label="suite size for the spread")''',
'''                b_n = gr.Slider(20, 600, 100, step=20,
                                label="conversations per arm - what you would buy")
                b_r = gr.Slider(1, 10, 3, step=1,
                                label="human raters per conversation")
                b_sd = gr.Slider(0.4, 1.6, 0.9, step=0.05,
                                 label="how much raters disagree (Likert sd)")
                b_ep = gr.Slider(20, 60, 40, step=10,
                                 label="suite size used to measure the spread")''')

# ------------------------------------------------ the missing bands tab -----
anchor = '        with gr.Tab("What\'s real here"):'
assert anchor in s, "What's real tab not found"
BANDS_TAB = '''        with gr.Tab("How to read the numbers"):
            gr.Markdown(
                "### Every number here has a scale, and the scale has a source\\n\\n"
                "A metric without a band is a number nobody can act on. But half "
                "of these cannot honestly carry a fixed threshold, and "
                "pretending otherwise is how a dashboard starts lying.\\n\\n"
                "- **Absolute** - means the same thing on any suite. \\"Ninety "
                "per cent of requests still honoured\\" is ninety per cent "
                "whoever is measuring.\\n"
                "- **Anchored** - a distance from an ideal that was defined for "
                "one action space. A simulated policy reaches 0.96 and the same "
                "real model measured through a transcript reaches 0.25; neither "
                "is \\"good\\" on its own. These are read against the range the "
                "suite can actually reach, drawn as a bar under the name in the "
                "release gate.\\n"
                "- **Conventional** - thresholds from psychometrics rather than "
                "from this package, cited rather than invented.\\n\\n"
                "Getting this distinction wrong is the same mistake as anchoring "
                "the environment's break-even point on simulated policies, which "
                "put every real conversation on the failure floor and left the "
                "suite with no discriminative power at all.")
            gr.HTML(reading_panel([
                "feedback uptake @10", "conversation failure rate",
                "left-over distress", "calibration", "panel: perceived empathy",
                "judge reliability", "substitution ratio"]))

'''
s = s.replace(anchor, BANDS_TAB + anchor)

io.open(p, "w", encoding="utf-8").write(s)
import ast
ast.parse(io.open(p, encoding="utf-8").read())
print("legends + orientation + bands tab eklendi, syntax OK")
