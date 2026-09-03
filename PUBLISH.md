# Publishing

## Live already

- **Hugging Face Space (public, static):**
  https://huggingface.co/spaces/renderfy/voicehaul
  Direct: https://renderfy-voicehaul.static.hf.space/
  Update it by editing `_web_template.html`, then:

      py -3 _export_web.py
      py -3 -c "tpl=open('_web_template.html',encoding='utf-8').read(); data=open('web_data.json',encoding='utf-8').read().replace('</','<\/'); out=tpl.replace('__DATA__',data); open('voicehaul_report.html','w',encoding='utf-8').write(out); open('docs/index.html','w',encoding='utf-8').write(out)"

  then copy `voicehaul_report.html` to `index.html` in the Space clone and push.

## GitHub — two commands, run these yourself

The repo is committed locally and ready. Create the empty repo at
https://github.com/new (name: `voicehaul`, **public**, no README, no licence,
no .gitignore — this repo already has all three), then:

    cd voicehaul
    git remote add origin https://github.com/vahit19/voicehaul.git
    git push -u origin main

## GitHub Pages — a second public URL, free

Repo → Settings → Pages → Source "Deploy from a branch", branch `main`,
folder `/docs`. Live in about a minute at https://vahit19.github.io/voicehaul/

`docs/index.html` is the same report the Space serves.

## Colab

Works as soon as the repo is public:

    https://colab.research.google.com/github/vahit19/voicehaul/blob/main/VoiceHaul_demo.ipynb

## Streamlit version

`space/` holds a Streamlit build of the same report that recomputes the whole
suite on every parameter change. It runs locally:

    cd space
    pip install -r requirements.txt
    streamlit run app.py

It is **not** deployed: Hugging Face now requires a PRO subscription to host
Docker or Gradio Spaces on free CPU (only `static` is free). The static Space
is arguably better for a cold link anyway — it loads instantly, where a free
Streamlit Space sleeps and shows a 30-second wake-up screen.
