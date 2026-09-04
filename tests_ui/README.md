# Testing the page, not just the package

`test_voicehaul.py` checks the library. These two check the thing a visitor
actually touches, against the deployed Space.

    pip install playwright && playwright install chromium
    python tests_ui/test_journey.py https://renderfy-voicehaul-app.hf.space/
    PYTHONPATH=. python tests_ui/test_numbers.py https://renderfy-voicehaul-app.hf.space/

**test_journey.py — mechanical.** Real clicks and keystrokes: open the page,
change both dropdowns, press the button, paste a transcript, run the gate,
download a file. Twenty-four assertions that the controls do something.

It exists because the page once shipped with its tab bar 1671px below the
fold and zero input elements in the initial DOM. It constructed without
error, screenshotted fine, and was unusable. Layout is not correctness, and
neither is "it rendered".

**test_numbers.py — correctness.** Downloads the artifact the page itself
serves, then recomputes every figure the page prints:

* the ratio on every row equals Spearman-Brown inverted, and round-trips
* the estimator is within 0.15 of the ground truth on every segment
* every row carries a bootstrap interval
* no dollar figure is quoted from an interval that does not clear the floor
* the transcript tab reports exactly what `transcript.analyse` computes for
  the same text, and invents nothing when the caller made no request

The last two are the ones that matter. A page can pass every mechanical check
and still print a number that is wrong, and a number that is wrong is worse
than a blank space, because it is believed.
