"""clean_spoken_text() is safe to call from several threads at once.

NeMo's Normalizer builds one TokenParser in __init__ and sets state on it
before each parse, so concurrent normalize() calls corrupt each other
(T441983). The service shares one Normalizer across FastAPI's worker
threads, which made /sections fail on articles that succeeded on their own.

The assertion is output equality, not merely "no exception": if the
interleaved parser state happens to be valid, the parse succeeds against
the wrong text and returns wrongly normalized output silently.

Integration test: needs the real nemo_text_processing and its grammar
cache, so it skips where NeMo is unavailable (CI has no grammar cache).
"""

import concurrent.futures

import pytest

pytest.importorskip("nemo_text_processing.text_normalization.normalize")

from src.models.tts_section_generator.tts_generator.text import (
    clean_spoken_text,
    init_nemo,
    nemo_available,
)

# Long, distinct, list-shaped text. Both properties are required:
#
#   * DISTINCT per thread -- identical inputs pass even under corruption,
#     since every thread corrupts alike;
#   * REALISTIC shape -- dense with numbers, dates and units, which is what
#     makes NeMo's tagger emit long token strings and widens the race
#     window. Short clean prose does not reproduce it.
TEXTS = [
    (
        "The line opened on 3 May 1920 and was extended in 1929, 1956 and 1972. "
        "It carries about 12,400 passengers each day over 8.7 km of track. "
        "Services run every 5 minutes between 06:00 and 21:00, and every "
        "10 minutes thereafter. The depot was rebuilt in 1998 at a cost of "
        "DM 42 million, and a further EUR 3.5 million was spent in 2004."
    ),
    (
        "The building rose 283 m above street level and held 71 floors when it "
        "was completed in 1931. It cost USD 14 million, covered 102,000 m2 of "
        "floor space, and was sold in 1998 for USD 150 million. A renovation "
        "between 2016 and 2019 replaced 6,400 windows and 23 lifts."
    ),
    (
        "The route runs 147 km from the junction at 51 deg 28 min north to the "
        "coast, climbing 1,200 m over 14 portages. Paddlers typically take "
        "between 6 and 9 days, covering 18 km per day. The survey of 1873 "
        "recorded 42 rapids, of which 11 were rated class III or higher."
    ),
    (
        "The district had 164,000 residents in 2021, up from 158,000 in 2011. "
        "Around 23 per cent were under 18, and 14.5 per cent were over 65. "
        "It covers 44.9 km2 and contains 37 schools, 9 libraries and "
        "3 hospitals built between 1894 and 1978."
    ),
]


@pytest.fixture(scope="module")
def normalize():
    init_nemo()
    if not nemo_available():
        pytest.skip("NeMo normalizer unavailable (no grammar cache)")
    return clean_spoken_text


def test_concurrent_normalization_is_stable_over_repeats(normalize):
    """The race is intermittent, so one clean pass is weak evidence."""
    expected = {text: normalize(text) for text in TEXTS}
    for _ in range(3):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(normalize, TEXTS * 2))
        assert results == [expected[text] for text in TEXTS * 2]
