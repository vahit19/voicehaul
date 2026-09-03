---
title: VoiceHaul
emoji: 🎚️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Long-horizon evaluation and judge substitution for voice AI
---

# VoiceHaul

**Long-horizon evaluation, failure-onset diagnosis and judge-substitution
analysis for empathic voice agents.**

A turn-level rating tells you whether a response *sounded* right. It cannot tell
you whether a model still honours what the caller asked for twenty turns ago,
whether its calibration decays as a session runs long, whether it is regulating
the caller's affect or mirroring it back, or which turn broke a call that ended
badly.

Everything on the page runs live. The harness itself is pure standard library;
gradio and plotly are here only to draw.

- **Release gate** — should this candidate ship? Welch tests with conversations
  as the unit, Holm-corrected, and an explicit statement of what the sample size
  could not resolve.
- **Watch a conversation** — one call, turn by turn.
- **Which turn broke it** — fault injection with ground truth, and a diagnostic
  that abstains rather than name a wrong turn.
- **Rating budget** — how many conversations, and which ones.
- **Judge substitution** — when can an automated rater replace a human one? Per
  dimension and per caller segment, with the estimator checked against ground
  truth.

Source: [github.com/vahit19/voicehaul](https://github.com/vahit19/voicehaul) ·
Static report: [vahit19.github.io/voicehaul](https://vahit19.github.io/voicehaul/)

Vahit Feryad — Apache-2.0
