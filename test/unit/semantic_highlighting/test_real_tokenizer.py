"""Checks boundaries.py against the tokenizer the service actually loads.

Everything in test_boundaries.py supplies `word_ids()` by hand, because the CI
image has no transformers. That leaves one assumption untested: that a fast
mDeBERTa-v3 tokenizer reports what `word_spans` reads it to report. This module
tests exactly that, against the real `timpal0l/mdeberta-v3-base-squad2`
tokenizer -- weights are not needed, since word alignment is a tokenizer
property, so the whole module runs in seconds on a CPU.

It is skipped unless ``SH_REAL_MODEL=1``, which also turns off the torch and
transformers stubs in conftest.py. Run it from the repo root inside the service
image, which already carries transformers:

    docker run --rm --user "$(id -u):$(id -g)" \
      -v "$PWD":/srv/app:ro -w /srv/app \
      -v "$HOME/.cache/huggingface":/hf:ro -e HF_HOME=/hf \
      -e USER="$USER" -e HOME=/tmp -e SH_REAL_MODEL=1 \
      -e PYTHONPATH=/srv/overlay/lib/python3.12/site-packages:/srv/venv/lib/python3.12/site-packages:/srv/app \
      --entrypoint python semantic-highlighting:prod \
      -m pytest test/unit/semantic_highlighting/test_real_tokenizer.py -q

The first run downloads the tokenizer (16 MB); after that ``HF_HUB_OFFLINE=1``
works. ``SH_TOKENIZER`` overrides which checkpoint is loaded, so the same
assertions can be pointed at a replacement checkpoint before it is deployed --
which is the other reason this is a test rather than a one-off script.
"""

import os

import pytest

from src.models.semantic_highlighting.model_server.boundaries import (
    _has_no_space_script,
    answer_span,
    word_spans,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SH_REAL_MODEL") != "1",
    reason="needs the real tokenizer; set SH_REAL_MODEL=1 to run",
)

CHECKPOINT = os.environ.get("SH_TOKENIZER", "timpal0l/mdeberta-v3-base-squad2")

QUESTION = "who designed it"
EIFFEL = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It is named after the engineer Gustave Eiffel, whose "
    "company designed and built the tower from 1887 to 1889 as the "
    "centrepiece of the World Exposition."
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(CHECKPOINT)


def encode(tokenizer, question, context):
    """Tokenize one pair the way `qa.answer_question_batch` does."""
    return tokenizer(
        [question],
        [context],
        return_tensors=None,
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=384,
        padding=True,
    )


def spans_of(tokenizer, question, context):
    """The `word_spans` output for one pair, with the encoding beside it."""
    enc = encode(tokenizer, question, context)
    offsets, sequence_ids = enc["offset_mapping"][0], enc.sequence_ids(0)
    return enc, word_spans(enc.word_ids(0), sequence_ids, offsets)


def token_at(enc, context, text):
    """Index of the first context token whose offsets cover exactly ``text``."""
    for i, (sequence, offset) in enumerate(
        zip(enc.sequence_ids(0), enc["offset_mapping"][0])
    ):
        if sequence == 1 and context[offset[0] : offset[1]] == text:
            return i
    raise AssertionError(f"no context token spans {text!r}")


def token_covering(enc, index):
    """Index of the context token whose offsets cover character ``index``.

    A word can repeat -- "Eiffel" and the comma after it both occur twice in
    the passage -- so a test about one of them anchors on where it is.
    """
    for i, (sequence, offset) in enumerate(
        zip(enc.sequence_ids(0), enc["offset_mapping"][0])
    ):
        if sequence == 1 and offset[0] <= index < offset[1]:
            return i
    raise AssertionError(f"no context token covers character {index}")


def widen(enc, spans, context, first, last):
    """Feed the raw token span from ``first`` to ``last`` through answer_span."""
    offsets = enc["offset_mapping"][0]
    start, end = answer_span(
        context,
        int(offsets[first][0]),
        int(offsets[last][1]),
        spans[first],
        spans[last],
    )
    return context[start:end]


# --- what the tokenizer reports ---------------------------------------------


def test_the_service_refuses_a_slow_tokenizer_and_this_one_is_fast(tokenizer):
    # qa.load_model raises unless this holds: offsets and word ids are the
    # Rust tokenizer's, and the decoder cannot map an answer back without them.
    assert tokenizer.is_fast


def test_word_ids_are_reported_at_all(tokenizer):
    enc = encode(tokenizer, QUESTION, EIFFEL)
    word_ids = enc.word_ids(0)
    # `word_spans` falls back to raw token offsets when this is None, which is
    # safe but silently drops the correction.
    assert word_ids is not None
    assert any(word is not None for word in word_ids)


def test_special_and_padding_tokens_carry_no_word(tokenizer):
    enc = tokenizer(
        [QUESTION, QUESTION],
        [EIFFEL, "Bananas are a tropical fruit."],
        return_tensors=None,
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=384,
        padding=True,
    )
    # The short item is padded out to the long one. Padding must report no
    # sequence and no word, or `word_spans` would union it into a real word.
    sequence_ids, word_ids = enc.sequence_ids(1), enc.word_ids(1)
    padding = [i for i, s in enumerate(sequence_ids) if s is None]
    assert padding, "expected the short item to be padded"
    assert all(word_ids[i] is None for i in padding)


def test_word_ids_restart_at_zero_for_the_context(tokenizer):
    """The question and the context both number their words from zero.

    `word_spans` keys its bounds by word id, so it would merge a question word
    into a context word of the same id if it did not filter on sequence 1
    first. Both of its loops do; this is what makes that filter load-bearing.
    """
    enc = encode(tokenizer, QUESTION, EIFFEL)
    sequence_ids, word_ids = enc.sequence_ids(0), enc.word_ids(0)
    question_words = {w for s, w in zip(sequence_ids, word_ids) if s == 0}
    context_words = {w for s, w in zip(sequence_ids, word_ids) if s == 1}
    assert 0 in question_words and 0 in context_words

    _, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    # No question token gets a span, and the first context word is "The".
    assert all(spans[i] is None for i, s in enumerate(sequence_ids) if s != 1)
    assert EIFFEL[slice(*spans[token_at(enc, EIFFEL, "The")])] == "The"


def test_a_word_span_covers_every_token_of_that_word(tokenizer):
    """The union is a superset of each of its tokens, and stays in the context."""
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    offsets, sequence_ids = enc["offset_mapping"][0], enc.sequence_ids(0)
    checked = 0
    for i, sequence in enumerate(sequence_ids):
        if sequence != 1 or spans[i] is None:
            continue
        start, end = int(offsets[i][0]), int(offsets[i][1])
        if end <= start:
            continue
        assert spans[i][0] <= start and end <= spans[i][1]
        assert 0 <= spans[i][0] < spans[i][1] <= len(EIFFEL)
        checked += 1
    assert checked > 20


def test_the_union_agrees_with_word_to_chars(tokenizer):
    """What `word_spans` builds by hand is what the encoding would report.

    `word_spans` unions the offsets of a word's tokens rather than calling
    `word_to_chars`, so that boundaries.py takes plain lists and can be tested
    without transformers. This is the check that the shortcut costs nothing --
    including at a truncation boundary, where a word can be cut in half.
    """
    long_context = (EIFFEL + " ") * 6
    differed = []
    for max_length in range(12, 200):
        enc = tokenizer(
            [QUESTION],
            [long_context],
            return_tensors=None,
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=max_length,
            padding=False,
        )
        sequence_ids, word_ids = enc.sequence_ids(0), enc.word_ids(0)
        spans = word_spans(word_ids, sequence_ids, enc["offset_mapping"][0])
        for i, sequence in enumerate(sequence_ids):
            if sequence != 1 or spans[i] is None:
                continue
            reported = enc.word_to_chars(0, word_ids[i], sequence_index=1)
            if spans[i] != (reported.start, reported.end):
                differed.append((max_length, i, spans[i], reported))
    assert not differed, differed[:5]


def test_a_word_span_carries_the_space_in_front_of_its_word(tokenizer):
    """Stated because it looks like a bug and is not.

    A SentencePiece token opens a word with the space before it, so the union
    does too. `_align_to_words` stops at a separator, so the space never
    reaches a highlight -- see `test_widening_stops_at_punctuation`.
    """
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    piece = token_at(enc, EIFFEL, "piece")
    assert EIFFEL[slice(*spans[piece])] == " centrepiece"
    assert widen(enc, spans, EIFFEL, piece, piece) == "centrepiece"


def test_this_tokenizer_really_does_split_centrepiece(tokenizer):
    """The premise of the whole correction, stated as a test.

    If a future checkpoint stops splitting this word, the alignment code is not
    wrong -- but this test is the record of why it exists.
    """
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    piece = token_at(enc, EIFFEL, "piece")
    assert EIFFEL[slice(*spans[piece])].strip() == "centrepiece"


# --- what the correction does with it ----------------------------------------


def test_an_edge_inside_a_word_widens_to_the_whole_word(tokenizer):
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    piece, world = token_at(enc, EIFFEL, "piece"), token_at(enc, EIFFEL, " World")
    assert widen(enc, spans, EIFFEL, piece, world) == "centrepiece of the World"


def test_widening_stops_at_punctuation(tokenizer):
    """`Gustave Eiffel`, not the pipeline's `" Gustave Eiffel,"`."""
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    name = EIFFEL.index("Gustave Eiffel,")
    first = token_covering(enc, name)
    last = token_covering(enc, EIFFEL.index("Eiffel,", name) + len("Eiffel") - 1)
    # The comma after the name is inside the tokenizer's word " Eiffel,", so
    # the word span offers it to the widening and the widening must decline.
    assert EIFFEL[slice(*spans[last])] == " Eiffel,"
    assert widen(enc, spans, EIFFEL, first, last) == "Gustave Eiffel"


def test_a_hyphen_stops_the_widening(tokenizer):
    """`iron`, not `wrought-iron`: the hyphen is a separator."""
    enc, spans = spans_of(tokenizer, QUESTION, EIFFEL)
    iron = token_at(enc, EIFFEL, "iron")
    assert EIFFEL[slice(*spans[iron])] == " wrought-iron"
    assert widen(enc, spans, EIFFEL, iron, iron) == "iron"


# --- the scripts the gate exists for -----------------------------------------

# One paragraph of the same content per script. The point of each is the size of
# a tokenizer "word", which is a whitespace-delimited run: in a script that
# writes without spaces that run is a whole clause, and widening an edge to it
# would highlight the clause.
NO_SPACE = {
    "ja": (
        "誰が設計しましたか",
        "エッフェル塔は、フランスのパリにあるシャン・ド・マルス公園に立つ錬鉄製の格子塔です。"
        "1887年から1889年にかけて万国博覧会の目玉として建設され、"
        "技術者のギュスターヴ・エッフェルの会社が設計しました。",
    ),
    "zh": (
        "谁设计的",
        "埃菲尔铁塔是位于法国巴黎战神广场的一座锻铁格构塔。它以工程师古斯塔夫·埃菲尔的名字命名，"
        "他的公司在1887年至1889年间为世界博览会设计并建造了这座塔。",
    ),
    "th": (
        "ใครเป็นผู้ออกแบบ",
        "หอไอเฟลเป็นหอคอยโครงเหล็กดัดตั้งอยู่บนช็องเดอมาร์สในกรุงปารีส ประเทศฝรั่งเศส "
        "ตั้งชื่อตามวิศวกรกุสตาฟ ไอเฟล ซึ่งบริษัทของเขาออกแบบและสร้างหอคอยนี้ระหว่างปี 1887 ถึง 1889",
    ),
}

SPACED = {
    "ar": (
        "من صممه",
        "برج إيفل هو برج شبكي من الحديد المطاوع يقع في حديقة شان دو مارس في باريس بفرنسا. "
        "سمي على اسم المهندس غوستاف إيفل الذي صممت شركته البرج وبنته بين عامي 1887 و1889.",
    ),
    "hi": (
        "इसे किसने डिज़ाइन किया",
        "एफिल टॉवर पेरिस, फ़्रांस में शैंप-दे-मार्स पर स्थित एक जालीदार लोहे का टॉवर है। "
        "इसका नाम इंजीनियर गुस्ताव एफिल के नाम पर रखा गया, जिनकी कंपनी ने 1887 से 1889 के बीच इसे बनाया।",
    ),
    "ko": (
        "누가 설계했나요",
        "에펠탑은 프랑스 파리의 샹 드 마르스 공원에 세워진 연철 격자 구조의 탑이다. "
        "1887년부터 1889년까지 만국 박람회를 위해 이 탑을 설계하고 건설한 기술자 귀스타브 에펠의 이름을 따서 지어졌다.",
    ),
}


def widest_word(tokenizer, question, context):
    """The longest tokenizer word in ``context``, as (text, character count)."""
    enc, spans = spans_of(tokenizer, question, context)
    words = {s for s in spans if s is not None}
    assert words
    widest = max(words, key=lambda s: s[1] - s[0])
    return context[widest[0] : widest[1]], widest[1] - widest[0]


@pytest.mark.parametrize("script", sorted(NO_SPACE))
def test_a_word_is_a_whole_clause_where_nothing_separates_words(tokenizer, script):
    """This is what the gate is for, measured rather than assumed."""
    question, context = NO_SPACE[script]
    text, length = widest_word(tokenizer, question, context)
    # A "word" here runs to tens of characters -- a clause, not a word.
    assert length > 40
    # And the gate catches it, so `_align_to_words` leaves the edge alone.
    assert _has_no_space_script(text)


@pytest.mark.parametrize("script", sorted(SPACED))
def test_a_spaced_script_gets_ordinary_words_and_aligns(tokenizer, script):
    question, context = SPACED[script]
    text, length = widest_word(tokenizer, question, context)
    assert length <= 20
    assert not _has_no_space_script(text)


@pytest.mark.parametrize("script", sorted(NO_SPACE))
def test_the_gate_leaves_the_model_token_edge_alone(tokenizer, script):
    """A mid-clause edge must not grow to the clause."""
    question, context = NO_SPACE[script]
    enc, spans = spans_of(tokenizer, question, context)
    sequence_ids = enc.sequence_ids(0)
    context_tokens = [i for i, s in enumerate(sequence_ids) if s == 1]
    # Take an edge in the middle of the passage, where a clause surrounds it.
    middle = context_tokens[len(context_tokens) // 2]
    highlighted = widen(enc, spans, context, middle, middle)
    word = spans[middle]
    if word is not None and _has_no_space_script(context[word[0] : word[1]]):
        # At most a grapheme cluster wider than the token the model chose.
        offsets = enc["offset_mapping"][0]
        raw = int(offsets[middle][1]) - int(offsets[middle][0])
        assert len(highlighted) <= raw + 2
        assert len(highlighted) < (word[1] - word[0])
