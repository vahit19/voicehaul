"""Renders docs/index.html from the template and the exported data.

Kept separate from _export_web.py so the page can be rebuilt after an edit to
the template without recomputing every number.
"""
import io

tpl = io.open("_web_template.html", encoding="utf-8").read()
data = io.open("web_data.json", encoding="utf-8").read().replace("</", "<\/")
out = tpl.replace("__DATA__", data)
for path in ("docs/index.html", "voicehaul_report.html"):
    io.open(path, "w", encoding="utf-8").write(out)
print("wrote docs/index.html ({} KB)".format(len(out) // 1024))
