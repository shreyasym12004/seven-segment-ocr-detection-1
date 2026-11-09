import numpy as np
import pytest

from sevenseg.pipeline import read_image
from sevenseg.preprocess import load_image
from sevenseg.recognize import PATTERNS, TABLE, SEGS
from sevenseg.synth import render


def test_pattern_table_unique_and_complete():
    assert {c for c in TABLE.values()} == set("0123456789")
    for ch, segs in PATTERNS.items():
        assert TABLE[tuple(int(s in segs) for s in SEGS)] == ch


@pytest.mark.parametrize("text", ["57901", "92312.5", "245.9", "04050", "6308.0", "1.2345"])
@pytest.mark.parametrize("kw", [{}, {"italic": 0.2}, {"tilt": 0.6}])
def test_reads_clean_synthetic(text, kw):
    rng = np.random.default_rng(0)
    assert read_image(render(text, rng, **kw))["raw_text"] == text


def test_meter_03_finds_all_display_digits_and_decimal():
    img = load_image("samples/meter_03.png")
    result = read_image(img)
    assert sum(ch.isdigit() for ch in result["raw_text"]) == 6
    assert "." in result["raw_text"]


def test_meter_01_recovers_low_contrast_digit_row():
    img = load_image("samples/meter_01.png")
    result = read_image(img)
    assert result["raw_text"] == "2927.40"
    assert result["detected_reading"] == 2927.4


def test_meter_04_ignores_wide_unit_mark_after_digits():
    img = load_image("samples/meter_04.png")
    result = read_image(img)
    assert sum(ch.isdigit() for ch in result["raw_text"]) == 5
