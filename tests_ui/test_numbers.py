"""Check that the numbers on the page are the right numbers.

The mechanical journey proves a button responds. This proves the response is
correct: every figure the UI prints is recomputed here from the library and
from the artifact the page itself serves, and compared.

Run with the repo on PYTHONPATH so the same functions the app uses are
importable.
"""
import json
import re
import sys

from playwright.sync_api import sync_playwright

from voicehaul.judge import substitution_ratio, spearman_brown
from voicehaul.cost import SUBSTITUTION_FLOOR, CostModel, estimate as cost_estimate
from voicehaul.transcript import analyse as analyse_transcript

URL = sys.argv[1]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + name +
          (" | " + detail if detail else ""), flush=True)


def field(txt, label):
    m = re.search(re.escape(label) + r"\s*\n\s*([^\n]+)", txt, re.I)
    return m.group(1).strip() if m else None


MINE = ("Caller: I have called four times about this charge and nobody calls back.\n"
        "Agent: I am so sorry, I really am sorry you have had to chase us like this.\n"
        "Caller: Stop apologising. Just fix it.\n"
        "Agent: I apologise for the delay. I will raise it now and I am sorry again.\n"
        "Caller: I told you to stop apologising.\n"
        "Agent: Understood. The refund is raised and billing will call you Tuesday.\n")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.set_default_timeout(60000)
    pg.goto(URL, wait_until="load")
    pg.wait_for_selector("button:has-text('Measure it')", state="visible")
    pg.wait_for_timeout(3500)

    # ------------------------------------------------------------------
    # 1. The artifact the page serves is the artifact the page displays.
    # ------------------------------------------------------------------
    print("\n=== 1. Kartlar artefaktla tutarli mi ===", flush=True)
    link = pg.locator("a[download], a[href*='/file']").first
    href = link.get_attribute("href")
    resp = pg.request.get(href if href.startswith("http")
                          else URL.rstrip("/") + href)
    art = json.loads(resp.body().decode("utf-8"))
    check("sayfanin sundugu artefakt indi",
          "rows" in art, "sema=%s satir=%d" % (art.get("schema"),
                                               len(art.get("rows", []))))

    def row_for(dim, seg):
        for r in art["rows"]:
            if r["dimension"] == dim and r["segment"] == seg:
                return r
        return None

    shown = pg.inner_text("body")
    r0 = row_for("actual_help", "all")
    ui_ratio = field(shown, "1 JUDGE RATING IS WORTH")
    check("varsayilan oran artefakttaki deger",
          ui_ratio == "%.2f" % r0["ratio_estimated"],
          "UI=%s artefakt=%.2f" % (ui_ratio, r0["ratio_estimated"]))

    ui_rho = field(shown, "JUDGE RELIABILITY")
    check("varsayilan rho artefakttaki deger",
          ui_rho == "%.2f" % r0["rho_judge_estimated"],
          "UI=%s artefakt=%.2f" % (ui_rho, r0["rho_judge_estimated"]))

    # ------------------------------------------------------------------
    # 2. The ratio is what Spearman-Brown inverted actually gives.
    # ------------------------------------------------------------------
    print("\n=== 2. Oran formulden yeniden hesaplandi ===", flush=True)
    bad = []
    for r in art["rows"]:
        mine = substitution_ratio(r["rho_judge_estimated"], r["rho_human_one"])
        if abs(mine - r["ratio_estimated"]) > 1e-6:
            bad.append((r["dimension"], r["segment"], mine, r["ratio_estimated"]))
    check("her satirda oran = k* formulu", not bad,
          "%d satir dogrulandi" % len(art["rows"]) if not bad
          else "uyusmayan: %s" % bad[:2])

    # Spearman-Brown must round-trip: k* raters of quality rho_h reach rho_g.
    rt = []
    for r in art["rows"]:
        k = r["ratio_estimated"]
        if k <= 0 or k > 1e4:
            continue
        got = spearman_brown(r["rho_human_one"], k)
        if abs(got - r["rho_judge_estimated"]) > 1e-4:
            rt.append((r["segment"], got, r["rho_judge_estimated"]))
    check("Spearman-Brown geri donusumlu", not rt,
          "uyusmayan: %s" % rt[:2] if rt else "tersine cevirme tutarli")

    # ------------------------------------------------------------------
    # 3. The estimator is close to the ground truth it can be checked against.
    # ------------------------------------------------------------------
    print("\n=== 3. Tahmin edici yer gerceklige karsi ===", flush=True)
    worst = max(art["rows"],
                key=lambda r: abs(r["rho_judge_estimated"] - r["rho_judge_true"]))
    err = abs(worst["rho_judge_estimated"] - worst["rho_judge_true"])
    check("rho hatasi her segmentte < 0.15", err < 0.15,
          "en kotu %s/%s = %+.3f" % (worst["dimension"], worst["segment"],
                                     worst["rho_judge_estimated"] -
                                     worst["rho_judge_true"]))

    # ------------------------------------------------------------------
    # 4. No dollar figure is quoted from an interval that does not clear it.
    # ------------------------------------------------------------------
    print("\n=== 4. Belirsizlik dogru sekilde kapiya konmus ===", flush=True)
    have_ci = [r for r in art["rows"] if r.get("ratio_hi", 0) > r.get("ratio_lo", 0)]
    check("her satirda bootstrap araligi var",
          len(have_ci) == len(art["rows"]),
          "%d/%d" % (len(have_ci), len(art["rows"])))

    ui_saving = field(shown, "SAVING PER YEAR")
    conf0 = (r0.get("ratio_hi", 0) > r0.get("ratio_lo", 0)
             and r0["ratio_lo"] >= SUBSTITUTION_FLOOR)
    check("varsayilan hucrede tasarruf $0 (aralik tabani gecmiyor)",
          (ui_saving == "$0") and not conf0,
          "UI=%s guvenli=%s alt=%.3f" % (ui_saving, conf0, r0.get("ratio_lo", 0)))

    # ------------------------------------------------------------------
    # 5. Tab 2: the page reports what the library computes for the same text.
    # ------------------------------------------------------------------
    print("\n=== 5. Kendi cagrim: UI vs kutuphane ===", flush=True)
    pg.locator("button[role=tab]:has-text('2 - Score your own call')").first.click()
    pg.wait_for_timeout(2500)
    ta = pg.locator("textarea").first
    ta.fill(MINE)
    btn = pg.locator("button:has-text('Measure this call')").first
    btn.evaluate("el => el.scrollIntoView({block:'center'})")
    pg.wait_for_timeout(300)
    btn.click()
    pg.wait_for_timeout(7000)
    t2 = pg.inner_text("body")

    rep = analyse_transcript(MINE)
    expect_pct = round(100 * rep.overall_uptake)
    m = re.search(r"(\d+)% of requests honoured", t2)
    check("UI'daki yuzde kutuphaneninkiyle ayni",
          bool(m) and int(m.group(1)) == expect_pct,
          "UI=%s%% kutuphane=%d%%" % (m.group(1) if m else "?", expect_pct))

    viol = [i for i, _ in rep.flagged]
    check("isaretlenen tur sayisi ayni",
          len(rep.flagged) >= 1 and ("turn %d" % rep.flagged[0][0]) in t2,
          "kutuphane ilk isaret: turn %d" % (rep.flagged[0][0] if rep.flagged else -1))

    # a control: a call with no request at all must not invent one
    CLEAN = ("Caller: What is my balance?\n"
             "Agent: It is forty pounds, due on the ninth.\n"
             "Caller: Thanks.\n"
             "Agent: Any time.\n")
    ta.fill(CLEAN)
    btn.evaluate("el => el.scrollIntoView({block:'center'})")
    pg.wait_for_timeout(300)
    btn.click()
    pg.wait_for_timeout(6000)
    t3 = pg.inner_text("body")
    rep2 = analyse_transcript(CLEAN)
    check("istek yoksa uydurmuyor",
          rep2.overall_uptake is None and "never asked for a change" in t3.lower(),
          "kutuphane uptake=%s" % rep2.overall_uptake)


    # ------------------------------------------------------------------
    # 6. The cell that used to quote $7,074 must no longer quote anything.
    # ------------------------------------------------------------------
    print("\n=== 6. Ince segment artik rakam vermiyor ===", flush=True)
    pg.locator("button[role=tab]:has-text('1 - Can a judge')").first.click()
    pg.wait_for_timeout(2000)
    for aria, opt in (("which quality is being rated", "perceived empathy"),
                      ("which callers", "grieving claimant")):
        dd = pg.locator("input[aria-label='%s']" % aria)
        dd.evaluate("el => el.scrollIntoView({block:'center'})")
        pg.wait_for_timeout(300)
        dd.click()
        pg.wait_for_timeout(700)
        pg.locator("li[data-testid=dropdown-option][aria-label='%s']" % opt).first.click()
        pg.wait_for_timeout(700)
    mb = pg.locator("button:has-text('Measure it')").first
    mb.evaluate("el => el.scrollIntoView({block:'center'})")
    pg.wait_for_timeout(300)
    mb.click()
    pg.wait_for_timeout(7000)
    t6 = pg.inner_text("body")
    g = row_for("perceived_empathy", "grieving_claim")
    check("ince hucrede tasarruf $0", field(t6, "SAVING PER YEAR") == "$0",
          "UI=%s (nokta tahmini oran %.2f)" % (field(t6, "SAVING PER YEAR"),
                                               g["ratio_estimated"]))
    check("aralik sinirsiz oldugu soyleniyor",
          "unbounded" in t6, "ratio_hi=%.0f" % g["ratio_hi"])
    check("kac tur gerektigi soyleniyor",
          "rated turns on this segment" in t6)
    check("eski $7,074 rakami sayfadan kalkti", "7,074" not in t6)

    b.close()

print("\n" + "=" * 60)
bad = [r for r in results if not r[1]]
print("DOGRULAMA: %d kontrol | %d PASS | %d FAIL" %
      (len(results), len(results) - len(bad), len(bad)))
for n_, _, d in bad:
    print("   FAIL:", n_, "|", d)
sys.exit(1 if bad else 0)
