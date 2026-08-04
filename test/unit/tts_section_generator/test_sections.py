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


IPA_FIXTURE = """
<html><body>
<section data-mw-section-id="0">
  <p>Władysław II Jagiełło
  <span class="IPA nowrap">[vwaˈdɨswaf jaˈɡʲɛwːɔ]</span>
  was the Grand Duke of Lithuania.</p>
</section>
</body></html>
"""


def test_ipa_pronunciation_guides_are_stripped():
    """v0 read IPA notation aloud (T424378); IPA spans are notation."""
    sec = extract_sections(IPA_FIXTURE)[0]
    assert "ˈ" not in sec.raw_text
    assert "ɨ" not in sec.raw_text
    assert "Władysław II Jagiełło" in sec.raw_text  # ł survives: content


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


PHONOS_FIXTURE = """
<html><body><section data-mw-section-id="0">
  <p>Jogaila (Lithuanian:
  <sup class="ext-phonos-attribution noexcerpt navigation-not-searchable">
  <a href="/wiki/File:x.ogg">ⓘ</a></sup>; born in Vilnius) was Grand Duke.</p>
</section></body></html>
"""


def test_phonos_attribution_icon_is_stripped():
    """The Phonos ⓘ 'listen' link is chrome, not speech (staging probe,
    T433594): rendered as a sibling of IPA spans, so the IPA strip
    doesn't catch it."""
    sec = extract_sections(PHONOS_FIXTURE)[0]
    assert "ⓘ" not in sec.raw_text
    assert "Jogaila" in sec.raw_text


GEO_FIXTURE = """
<html><body><section data-mw-section-id="0"><p>The memorial stands at
<span class="geo-default"><span class="geo-dms"><span class="latitude">40°41′21″N</span>
<span class="longitude">74°2′40″W</span></span></span><span
class="geo-multi-punct"> / </span><span class="geo-nondefault"><span
class="geo-dec">40.689°N 74.044°W</span></span> in the harbor, near the
old fort that guarded the approaches.</p></section></body></html>
"""


def test_geo_hidden_duplicate_is_stripped():
    """{{Coord}} emits the coordinate twice (DMS + CSS-hidden decimal twin
    + hidden ' / '); the extractor applies no CSS, so without the strip an
    inline coordinate is read twice with a stray slash."""
    sec = extract_sections(GEO_FIXTURE)[0]
    assert "40°41′21″N" in sec.raw_text
    assert "40.689" not in sec.raw_text
    assert " / " not in sec.raw_text


HIDDEN_BEGIN_FIXTURE = """
<html><body><section data-mw-section-id="20"><h2>Family tree</h2>
<div class="hidden-begin mw-collapsible" style="border:1px solid #667766;">
<div class="hidden-title"><b>Family tree of Jogaila/Wladyslaw II
Jagiello</b></div>
<div class="hidden-content mw-collapsible-content"><table><tr><td>
Gediminas b. c. 1275 d. 1341</td></tr></table></div>
</div>
<p>The dynasty he founded reigned over the union for nearly two hundred
years, shaping the region's politics and faith for generations to
come.</p></section></body></html>
"""


def test_hidden_begin_collapsed_content_is_stripped():
    """{{Hidden begin}}: collapsed-by-default supplementary content
    (family trees). Only its title bar used to leak, read aloud as an
    orphan caption (T433923: 75% WER, the slash spoken). The sibling
    prose must survive."""
    sec = extract_sections(HIDDEN_BEGIN_FIXTURE)[0]
    assert "Family tree of Jogaila" not in sec.raw_text
    assert "Gediminas" not in sec.raw_text
    assert "The dynasty he founded" in sec.raw_text
