"""Use the deployed page the way a first-time visitor would.

Every step is a real click or keystroke through Playwright. A control that
only looks clickable fails here exactly as it would for a person.
"""
import re
import sys
from playwright.sync_api import sync_playwright, expect

URL = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else "."
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + name +
          (" | " + detail if detail else ""), flush=True)


def card(page, label):
    """Read the big number under one of the three headline cards."""
    el = page.locator("div", has_text=re.compile("^" + label + "$", re.I))
    txt = page.inner_text("body")
    m = re.search(re.escape(label) + r"\s*\n\s*([^\n]+)", txt, re.I)
    return m.group(1).strip() if m else None



def tap(page, selector, label=""):
    """Click the way a person does, after putting the target mid-screen.

    Gradio's tab bar is sticky, so an element scrolled to the very top can sit
    underneath it. Centring first removes that, and keeps the click a real
    hit-tested mouse event rather than a dispatched DOM event.
    """
    loc = page.locator(selector).first
    loc.scroll_into_view_if_needed()
    loc.evaluate("el => el.scrollIntoView({block:'center', inline:'center'})")
    page.wait_for_timeout(350)
    loc.click()


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.set_default_timeout(45000)
    pg.goto(URL, wait_until="load")

    # Gradio streams the initial render; wait for a real control, not a timer.
    pg.wait_for_selector("button:has-text('Measure it')", state="visible")
    pg.wait_for_timeout(3000)

    print("\n=== SEKME 1: Can a judge replace your raters? ===", flush=True)
    t0 = pg.inner_text("body")
    check("acilista sonuc var (hic tiklamadan)",
          "judge rating is worth" in t0.lower(),
          "oran=%s" % card(pg, "1 JUDGE RATING IS WORTH"))
    check("varsayilan tasarruf $0",
          card(pg, "SAVING PER YEAR") == "$0",
          "okunan=%s" % card(pg, "SAVING PER YEAR"))
    check("varsayilan karar 'panel devredilemez'",
          "cannot be replaced" in t0)
    pg.screenshot(path=SHOTS + "/j1_open.png")

    # --- change the two dropdowns the way a person does -------------------
    dd1 = pg.locator("input[aria-label='which quality is being rated']")
    dd1.scroll_into_view_if_needed()
    dd1.evaluate("el => el.scrollIntoView({block:'center'})")
    pg.wait_for_timeout(350)
    dd1.click()
    pg.wait_for_timeout(600)
    opt = pg.locator("li[data-testid=dropdown-option]"
                     "[aria-label='perceived empathy']").first
    check("boyut menusu acildi ve secenek gorundu", opt.is_visible())
    opt.click()
    pg.wait_for_timeout(800)
    check("boyut 'perceived empathy' secildi",
          dd1.input_value() == "perceived empathy",
          "deger=%r" % dd1.input_value())

    dd2 = pg.locator("input[aria-label='which callers']")
    dd2.scroll_into_view_if_needed()
    dd2.evaluate("el => el.scrollIntoView({block:'center'})")
    pg.wait_for_timeout(350)
    dd2.click()
    pg.wait_for_timeout(600)
    opt2 = pg.locator("li[data-testid=dropdown-option]"
                      "[aria-label='grieving claimant']").first
    check("arayan menusu acildi ve secenek gorundu", opt2.is_visible())
    opt2.click()
    pg.wait_for_timeout(800)
    check("arayan 'grieving claimant' secildi",
          dd2.input_value() == "grieving claimant",
          "deger=%r" % dd2.input_value())

    tap(pg, "button:has-text('Measure it')")
    pg.wait_for_timeout(7000)
    t1 = pg.inner_text("body")
    ratio = card(pg, "1 JUDGE RATING IS WORTH")
    saving = card(pg, "SAVING PER YEAR")
    check("cikti degisti", t1 != t0)
    check("oran 1.56 oldu", (ratio or "").startswith("1.5"), "okunan=%s" % ratio)
    # The ratio on this cell is 1.56, but its interval runs to unbounded, so
    # the page must refuse to price it. Quoting a figure here is the bug.
    check("ince segmentte rakam verilmiyor", saving == "$0",
          "okunan=%s" % saving)
    check("neden verilmedigi yaziyor",
          "cannot back it up" in t1 and "unbounded" in t1)
    check("kac tur gerektigi soyleniyor",
          "rated turns on this segment" in t1)
    pg.screenshot(path=SHOTS + "/j2_measured.png")

    print("\n=== SEKME 2: Score your own call ===", flush=True)
    tap(pg, "button[role=tab]:has-text('2 - Score your own call')")
    pg.wait_for_timeout(2500)
    ta = pg.locator("textarea").first
    check("transkript kutusu goruniyor ve dolu geliyor",
          ta.is_visible() and len(ta.input_value()) > 100,
          "uzunluk=%d" % len(ta.input_value()))
    before2 = pg.inner_text("body")

    MINE = ("Caller: I have called four times about this charge and nobody calls back.\n"
            "Agent: I am so sorry, I really am sorry you have had to chase us like this.\n"
            "Caller: Stop apologising. Just fix it.\n"
            "Agent: I apologise for the delay. I will raise it now and I am sorry again.\n"
            "Caller: I told you to stop apologising.\n"
            "Agent: Understood. The refund is raised and billing will call you Tuesday.\n")
    ta.fill(MINE)
    pg.wait_for_timeout(600)
    check("kendi transkriptim yazildi",
          "called four times" in ta.input_value())
    tap(pg, "button:has-text('Measure this call')")
    pg.wait_for_timeout(7000)
    after2 = pg.inner_text("body")
    check("kendi cagrima cevap uretti", after2 != before2)
    check("istegin ihlal edildigi yakalandi",
          "apolog" in after2.lower(),
          "ciktida 'apolog' geciyor")
    pg.screenshot(path=SHOTS + "/j3_owncall.png")

    print("\n=== SEKME 3: Should this candidate ship? ===", flush=True)
    tap(pg, "button[role=tab]:has-text('3 - Should this candidate ship?')")
    pg.wait_for_timeout(2500)
    before3 = pg.inner_text("body")
    check("acilista bir karar goruniyor",
          "SHIP" in before3.upper() or "BLOCK" in before3.upper(),
          "BLOCK" if "BLOCK" in before3.upper() else "SHIP")
    tap(pg, "button:has-text('Run the gate')")
    pg.wait_for_timeout(25000)
    after3 = pg.inner_text("body")
    check("gate kosturuldu ve karar dondu",
          "SHIP" in after3.upper() or "BLOCK" in after3.upper())
    pg.screenshot(path=SHOTS + "/j4_gate.png")

    print("\n=== indirilebilir ciktilar ===", flush=True)
    links = pg.locator("a[download], a[href*='/file']")
    n = links.count()
    check("indirme baglantisi uretildi", n > 0, "adet=%d" % n)
    if n:
        href = links.first.get_attribute("href")
        resp = pg.request.get(href if href.startswith("http")
                              else URL.rstrip("/") + href)
        body = resp.body()
        check("ilk dosya gercekten iniyor",
              resp.ok and len(body) > 50,
              "http=%d %d bayt" % (resp.status, len(body)))

    print("\n=== alttaki bolumler ===", flush=True)
    body = pg.inner_text("body")
    for want in ("How to read these numbers", "How it works",
                 "What's real here", "The parts that did not fit"):
        check("bolum mevcut: " + want, want in body)

    b.close()

print("\n" + "=" * 58)
bad = [r for r in results if not r[1]]
print("TOPLAM %d kontrol | %d PASS | %d FAIL" %
      (len(results), len(results) - len(bad), len(bad)))
for n_, _, d in bad:
    print("   FAIL:", n_, "|", d)
sys.exit(1 if bad else 0)
