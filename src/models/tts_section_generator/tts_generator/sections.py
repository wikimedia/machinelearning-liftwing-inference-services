"""Section extraction from revision-pinned Parsoid HTML.

Successor to v0's wikipedia_utils.py, re-targeted from the wikipedia-api
section tree to Parsoid's structural markup. Parsoid wraps each section in
``<section data-mw-section-id="N">`` (0 = lead, >0 = editable heading
sections, <0 = template-generated), nesting subsections inside their
parents, so traversal is structural rather than heading-level arithmetic.

This module owns three contract-level definitions the rest of the system
depends on:

* ``section_id``: lowercase heading slug, ``-N`` ordinal suffix for
  duplicate headings, ``"lead"`` for the lead. Deterministic from heading
  text and document order, which means a renamed heading is a new
  section_id (a regeneration), an accepted property.
* the section blocklist (v0's, unchanged): blocklisted sections and their
  entire subtrees are excluded.
* per-section TEXT: the section element's own prose, excluding nested
  subsections (they are their own sections) and non-spoken noise
  (references, tables, figures, infoboxes, math, edit links, hatnotes).
"""

import logging
import re
from dataclasses import dataclass, field

from lxml import html as lxml_html

logger = logging.getLogger(__name__)

BLOCKLIST = {"see also", "references", "external links", "further reading", "notes"}

# Elements that must never reach the TTS voice. Tag-level strips first,
# then class-token strips (Parsoid and skin classes).
_STRIP_TAGS = {
    "style",
    "script",
    "table",
    "figure",
    "figcaption",
    "math",
    "audio",
    "video",
}
_STRIP_CLASS_TOKENS = {
    "mw-ref",  # Parsoid inline citation markers [1]
    "reference",
    "mw-references-wrap",
    "mw-editsection",
    "infobox",
    "navbox",
    "hatnote",  # "Main article: ..." disambiguation lines
    "dablink",
    "mwe-math-element",  # math renders as LaTeX alt text, unreadable aloud
    "noprint",
    "mw-empty-elt",
    "thumb",
    "gallery",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Section:
    """One valid, addressable section of a pinned revision."""

    section_id: str
    title: str
    level: int  # 1 = lead, 2..6 = heading level
    raw_text: str  # extracted prose, pre-normalization
    mw_section_id: int  # Parsoid's data-mw-section-id, kept for debugging


@dataclass
class _SlugCounter:
    seen: dict = field(default_factory=dict)

    def make(self, title: str) -> str:
        slug = _SLUG_RE.sub("-", title.lower()).strip("-") or "section"
        n = self.seen.get(slug, 0) + 1
        self.seen[slug] = n
        return slug if n == 1 else f"{slug}-{n}"


def _should_strip(el) -> bool:
    if not isinstance(el.tag, str):  # comments, processing instructions
        return True
    if el.tag in _STRIP_TAGS:
        return True
    classes = set((el.get("class") or "").split())
    return bool(classes & _STRIP_CLASS_TOKENS)


def _own_text(section_el) -> str:
    """Extract a section element's own prose.

    Works on a copy: drops nested <section> elements (subsections are
    emitted separately), drops the heading itself (spoken section titles
    are the client's concern, not part of the body audio), then drops
    noise elements and flattens to whitespace-normalized text.
    """
    el = lxml_html.fromstring(lxml_html.tostring(section_el))
    for nested in el.findall(".//section"):
        nested.drop_tree()
    for h in el.xpath(
        ".//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]"
    ):
        h.drop_tree()
    # Iterate a static list: drop_tree mutates the tree under us otherwise.
    for node in list(el.iter()):
        if node is el:
            continue
        if _should_strip(node):
            node.drop_tree()
    text = el.text_content()
    return re.sub(r"\s+", " ", text).strip()


def _heading_of(section_el) -> tuple[str, int] | None:
    """Return (title, level) from the section's own heading, if any.

    The heading is a direct-ish child; anything inside a nested <section>
    belongs to a subsection. Parsoid emits <h2>..<h6>; some skins wrap in
    <div class="mw-heading">, which xpath descent handles transparently.
    """
    for h in section_el.xpath(
        ".//*[self::h2 or self::h3 or self::h4 or self::h5 or self::h6]"
    ):
        # Reject headings that belong to a nested subsection.
        anc = h.getparent()
        crossed = False
        while anc is not None and anc is not section_el:
            if anc.tag == "section":
                crossed = True
                break
            anc = anc.getparent()
        if crossed:
            continue
        # Strip noise inside the heading itself (mw-editsection spans and
        # similar) BEFORE reading its text: "References<span
        # class=mw-editsection>edit</span>" must blocklist-match as
        # "References", not "Referencesedit".
        h_clean = lxml_html.fromstring(lxml_html.tostring(h))
        for node in list(h_clean.iter()):
            if node is not h_clean and _should_strip(node):
                node.drop_tree()
        title = re.sub(r"\s+", " ", h_clean.text_content()).strip()
        level = int(h.tag[1])
        return title, level
    return None


def extract_sections(html: str) -> list[Section]:
    """Extract all valid sections from Parsoid HTML, in document order.

    The lead (data-mw-section-id="0") is always first, titled "Lead" with
    section_id "lead", mirroring v0's convention. Blocklisted sections are
    skipped together with their entire subtree (v0 semantics). Sections
    with negative mw-section-ids (template-generated) are skipped.
    """
    doc = lxml_html.fromstring(html)
    body = doc.find("body") if doc.tag == "html" else doc
    if body is None:
        body = doc

    slugs = _SlugCounter()
    out: list[Section] = []

    def walk(el) -> None:
        for child in el:
            if not isinstance(child.tag, str):
                continue
            if child.tag != "section":
                # Non-section wrapper (rare in Parsoid output): descend.
                walk(child)
                continue

            try:
                mw_id = int(child.get("data-mw-section-id", "-999"))
            except ValueError:
                mw_id = -999

            if mw_id == 0:
                out.append(
                    Section(
                        section_id="lead",
                        title="Lead",
                        level=1,
                        raw_text=_own_text(child),
                        mw_section_id=0,
                    )
                )
                walk(child)  # lead never has subsections, but stay safe
                continue

            heading = _heading_of(child)
            if heading is None or mw_id < 0:
                logger.debug("Skipping section without heading (mw_id=%s)", mw_id)
                continue

            title, level = heading
            if title.lower() in BLOCKLIST:
                continue  # blocklist prunes the whole subtree: do not recurse

            out.append(
                Section(
                    section_id=slugs.make(title),
                    title=title,
                    level=level,
                    raw_text=_own_text(child),
                    mw_section_id=mw_id,
                )
            )
            walk(child)  # emit subsections as their own entries

    walk(body)
    return out


def find_section(sections: list[Section], section_id: str) -> Section | None:
    for s in sections:
        if s.section_id == section_id:
            return s
    return None
