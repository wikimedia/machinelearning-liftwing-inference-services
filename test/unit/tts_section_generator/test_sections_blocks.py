"""Section.block extraction: the same prose as raw_text, block by block.

Paragraph-level timing needs to know where each paragraph begins and ends
in the spoken text, and _own_text has flattened that away. These tests pin
the block list to the same pruning rules as raw_text so the two can never
diverge (a block that isn't the text actually synthesized would point
paragraph timings at the wrong words).
"""

from src.models.tts_section_generator.tts_generator.sections import (
    extract_sections,
    find_section,
)

FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <p>Earth is the third planet from the Sun.</p>
  <p>It is the only place known to harbor life.</p>
</section>
<section data-mw-section-id="1">
  <h2>History</h2>
  <p>Earth formed about 4.5 billion years ago.</p>
  <ul><li>Early life appeared in the oceans.</li></ul>
  <section data-mw-section-id="2">
    <h3>Early life</h3>
    <p>Life diversified.</p>
  </section>
</section>
</body></html>
"""


def test_lead_blocks_are_the_prose_paragraphs_in_order():
    lead = find_section(extract_sections(FIXTURE), "lead")
    assert lead.blocks == [
        "Earth is the third planet from the Sun.",
        "It is the only place known to harbor life.",
    ]


def test_parent_blocks_exclude_subsection_text():
    history = find_section(extract_sections(FIXTURE), "history")
    assert history.blocks == [
        "Earth formed about 4.5 billion years ago.",
        "Early life appeared in the oceans.",
    ]


NESTED_FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <ul><li><p>Outer list item with an inner paragraph.</p></li></ul>
</section>
</body></html>
"""


def test_nested_blocks_are_attributed_once_to_the_outermost():
    sec = extract_sections(NESTED_FIXTURE)[0]
    assert sec.blocks == ["Outer list item with an inner paragraph."]


NOISE_FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <p>Earth is a planet.<sup class="mw-ref"><a>[1]</a></sup></p>
</section>
</body></html>
"""


def test_blocks_are_pruned_of_noise_elements():
    sec = extract_sections(NOISE_FIXTURE)[0]
    assert sec.blocks == ["Earth is a planet."]


MIXED_FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <p>Intro paragraph.</p>
  <blockquote>A famous quote.</blockquote>
  <dl><dt>Term</dt><dd>Definition of the term.</dd></dl>
</section>
</body></html>
"""


def test_all_block_tags_are_captured_in_document_order():
    sec = extract_sections(MIXED_FIXTURE)[0]
    assert sec.blocks == [
        "Intro paragraph.",
        "A famous quote.",
        "Term",
        "Definition of the term.",
    ]


# ── raw_text / blocks divergence (content_sha256 safety) ────────────────────


ADJACENT_FIXTURE = (
    '<html><body><section data-mw-section-id="0">'
    "<p>One.</p><p>Two.</p>"
    "</section></body></html>"
)


def test_raw_text_flattens_adjacent_blocks_without_separator():
    """raw_text must stay text_content()-flattened, not " ".join(blocks).

    Adjacent blocks with no whitespace between them flatten to "One.Two."
    (no separator); deriving raw_text from blocks would insert a space and
    move content_sha256 for exactly these sections. This pins the
    pre-refactor output so the blocks refactor cannot silently change it.
    """
    sec = extract_sections(ADJACENT_FIXTURE)[0]
    assert sec.raw_text == "One.Two."


def test_joined_blocks_do_not_reproduce_raw_text():
    """The deliberate divergence that justifies blocks-as-sidecar.

    " ".join(blocks) must NOT reproduce raw_text: if it did, deriving
    raw_text from blocks would look safe, and this change would silently
    move content_sha256 for sections with adjacent blocks. The two are
    intentionally different descriptions of the same prose.
    """
    sec = extract_sections(ADJACENT_FIXTURE)[0]
    assert sec.blocks == ["One.", "Two."]
    assert " ".join(sec.blocks) == "One. Two."
    assert " ".join(sec.blocks) != sec.raw_text
