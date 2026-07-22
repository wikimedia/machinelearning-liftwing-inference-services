"""Section extraction tests against Parsoid-shaped fixtures."""

from src.models.tts_section_generator.tts_generator.sections import (
    extract_sections,
    find_section,
)

FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <table class="infobox"><tbody><tr><td>Infobox junk 12345</td></tr></tbody></table>
  <p>Earth is the third planet from the Sun.<sup class="mw-ref"><a>[1]</a></sup>
     It is the only place known to harbor life.</p>
  <figure><img src="x"/><figcaption>A photo caption</figcaption></figure>
</section>
<section data-mw-section-id="1">
  <h2 id="History">History</h2>
  <div class="hatnote">Main article: History of Earth</div>
  <p>Earth formed about 4.5 billion years ago.</p>
  <section data-mw-section-id="2">
    <h3 id="Early_life">Early life</h3>
    <p>Life appeared in the oceans.</p>
  </section>
</section>
<section data-mw-section-id="3">
  <h2 id="Notes_on_names">History</h2>
  <p>A second section that reuses the History heading.</p>
</section>
<section data-mw-section-id="4">
  <h2 id="References">References<span class="mw-editsection">edit</span></h2>
  <ol class="references"><li>Ref one</li></ol>
  <section data-mw-section-id="5">
    <h3 id="Refsub">Should be skipped with parent</h3>
    <p>Never emitted.</p>
  </section>
</section>
<section data-mw-section-id="6">
  <h2 id="Culture">Culture</h2>
  <p>Culture text with a table.</p>
  <table class="wikitable"><tbody><tr><td>tabular noise</td></tr></tbody></table>
</section>
</body></html>
"""

REFBEGIN_FIXTURE = """
<html><body>
<section data-mw-section-id="0"><p>Lead text long enough to pass the gate,
padded with more words to comfortably exceed the fifty character minimum.</p></section>
<section data-mw-section-id="1">
  <h2>Sources</h2>
  <div class="refbegin">
    <ul><li>Author, A. (1990). A Very Long Book Title. Publisher.</li>
    <li>Writer, B. (2001). Another Citation. Press.</li></ul>
  </div>
</section>
<section data-mw-section-id="2">
  <h2>Bibliography</h2>
  <p>Should never appear regardless of content: title is blocklisted.</p>
</section>
<section data-mw-section-id="3">
  <h2>Selected works</h2>
  <div class="div-col">
    <cite class="citation book">Composer, C. (1886). The Carnival of the
    Animals. Paris: Editions.</cite>
    <cite class="citation book">Composer, C. (1921). Another Work. Press.</cite>
  </div>
</section>
</body></html>
"""

INTERLINEAR_FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <p>The Greece runestones are about thirty Varangian Runestones.</p>
</section>
<section data-mw-section-id="1">
  <h4 id="U_518">U 518</h4>
  <p>This runestone is in the RAK style and is raised on the side of a
  rocky outcrop. The stone was made known by Richard Dybeck in several
  publications in the 1860s.</p>
  <div class="interlinear">
    <p lang="non">þurkir × uk × suin × þina × iftiʀ × irmn × trs</p>
    <p lang="non-Latn">Þorgeirr ok Sveinn þenna eptir Orm ok Ormulf</p>
  </div>
  <p>The inscription ends with a prayer in Old Norse.</p>
</section>
</body></html>
"""


def test_document_order_and_blocklist_subtree_pruning():
    ids = [s.section_id for s in extract_sections(FIXTURE)]
    assert ids == ["lead", "history", "early-life", "history-2", "culture"]


def test_lead_extraction_strips_infobox_refs_and_captions():
    lead = find_section(extract_sections(FIXTURE), "lead")
    assert lead.title == "Lead"
    assert lead.level == 1
    assert "third planet from the Sun" in lead.raw_text
    assert "harbor life" in lead.raw_text
    assert "Infobox junk" not in lead.raw_text
    assert "[1]" not in lead.raw_text
    assert "photo caption" not in lead.raw_text


def test_parent_text_excludes_subsection_text():
    history = find_section(extract_sections(FIXTURE), "history")
    assert "4.5 billion years" in history.raw_text
    assert "Life appeared" not in history.raw_text
    assert "Main article" not in history.raw_text


def test_subsection_is_its_own_entry_with_level():
    early = find_section(extract_sections(FIXTURE), "early-life")
    assert early.title == "Early life"
    assert early.level == 3
    assert "Life appeared in the oceans." == early.raw_text


def test_duplicate_heading_gets_ordinal_suffix():
    dup = find_section(extract_sections(FIXTURE), "history-2")
    assert dup.title == "History"
    assert "reuses the History heading" in dup.raw_text


def test_tables_and_editsection_stripped():
    culture = find_section(extract_sections(FIXTURE), "culture")
    assert "Culture text" in culture.raw_text
    assert "tabular noise" not in culture.raw_text
    refs_absent = find_section(extract_sections(FIXTURE), "references")
    assert refs_absent is None


def test_refbegin_bibliography_content_is_stripped_structurally():
    secs = {s.section_id: s for s in extract_sections(REFBEGIN_FIXTURE)}
    assert "bibliography" not in secs
    assert secs["sources"].raw_text == ""
    assert secs["selected-works"].raw_text == ""


def test_interlinear_gloss_apparatus_is_stripped():
    """Regression: interlinear-gloss template containers (div.interlinear)
    wrap Old Norse transliteration pairs that are linguistic apparatus, not
    prose. Before this fix, runestone U 518's lang="non" / lang="non-Latn"
    pairs survived into raw_text, got chunked, and phonemized past Kokoro's
    510-phoneme limit, crashing the isvc (pilot 502 synthesis_error)."""
    secs = {s.section_id: s for s in extract_sections(INTERLINEAR_FIXTURE)}
    u518 = secs["u-518"]
    assert "runestone is in the RAK style" in u518.raw_text
    assert "prayer in Old Norse" in u518.raw_text
    assert "þurkir" not in u518.raw_text
    assert "Þorgeirr" not in u518.raw_text
    assert "interlinear" not in u518.raw_text
