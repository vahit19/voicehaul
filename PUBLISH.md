# Where this lives

| | |
|---|---|
| source | https://github.com/vahit19/voicehaul |
| report | https://vahit19.github.io/voicehaul/ |
| mirror | https://huggingface.co/spaces/renderfy/voicehaul |
| notebook | [Colab](https://colab.research.google.com/github/vahit19/voicehaul/blob/main/VoiceHaul_demo.ipynb) |

## Regenerating the report

`docs/index.html` is the page GitHub Pages serves. It is generated, not edited:

    python _export_web.py          # recompute every number
    python _build_report.py        # template + data -> docs/index.html

Then commit and push. To mirror it on the Space, copy `docs/index.html` to
`index.html` in a clone of the Space repo and push there.

## Regenerating the numbers it shows

    python run_demo.py             # the simulated suite, plus the static chart
    python run_llm_gate.py --config configs/llm-prompt-change.yaml
    python run_substitution.py --source llm

The last two need a model. They try a local ollama first, then OpenRouter or
the Hugging Face router if a key is set. Responses are cached under
`.voicehaul-cache/`, so a second run is free and produces identical numbers.
