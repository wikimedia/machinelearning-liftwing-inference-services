"""Revision-pinned article fetch via the MediaWiki REST API.

This is the one component with no v0 ancestor: v0's wikipedia-api client
could only fetch the CURRENT revision, which is why v0 artifacts were not
revision-addressable. Here every fetch is pinned to an explicit rev_id, so
a generated artifact provably corresponds to the revision recorded in its
index entry, the property the whole storage/index design rests on.

Source of truth: Parsoid HTML from ``GET /w/rest.php/v1/revision/{id}/html``.
Parsoid (not raw wikitext) because templates arrive expanded, and not the
TextExtracts API because it cannot address old revisions. Parsoid also wraps
every section in ``<section data-mw-section-id="N">``, which makes section
segmentation structural rather than heuristic (see sections.py).
"""

import logging
import re

import requests
from requests.adapters import HTTPAdapter
from tts_generator.config import (
    FETCH_RETRIES,
    FETCH_TIMEOUT_S,
    MW_API_PROXY,
    USER_AGENT,
)
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_WIKI_ID_RE = re.compile(r"^([a-z]{2,12})wiki$")


class FetchError(Exception):
    """Upstream MediaWiki fetch failure, carrying the upstream HTTP status."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def rest_base(wiki_id: str) -> str:
    """Map a wiki_id ('enwiki') to its REST API base URL.

    Only the ``{lang}wiki`` -> ``{lang}.wikipedia.org`` family is supported
    in Phase 1; extending to sister projects means extending this map, not
    the callers.

    When TTS_GEN_MW_API_PROXY is set, the URL points at the envoy
    services-proxy (e.g. ``http://localhost:6500``) instead of hitting
    Wikipedia directly, and callers must also set the Host header via
    :func:`wiki_host`.
    """
    m = _WIKI_ID_RE.match(wiki_id)
    if not m:
        raise FetchError(f"Unsupported wiki_id {wiki_id!r}", status=400)
    if MW_API_PROXY:
        return f"{MW_API_PROXY}/w/rest.php/v1"
    return f"https://{m.group(1)}.wikipedia.org/w/rest.php/v1"


def wiki_host(wiki_id: str) -> str:
    """Host header the MW API proxy uses to route to the correct wiki."""
    m = _WIKI_ID_RE.match(wiki_id)
    if not m:
        raise FetchError(f"Unsupported wiki_id {wiki_id!r}", status=400)
    return f"{m.group(1)}.wikipedia.org"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=FETCH_RETRIES,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _headers(wiki_id: str) -> dict | None:
    """Return extra headers for requests through the envoy services-proxy."""
    if MW_API_PROXY:
        return {"Host": wiki_host(wiki_id)}
    return None


_shared_session = _session()


def fetch_revision_meta(wiki_id: str, rev_id: int) -> dict:
    """Fetch revision metadata (``/revision/{id}/bare``).

    Used to verify the caller's (page_id, rev_id) pairing before any
    generation work: a mismatch means the pipeline's bookkeeping is wrong,
    and generating under a mislabeled key would poison the index.

    Returns the parsed JSON, which includes ``page.id`` and ``timestamp``.
    """
    url = f"{rest_base(wiki_id)}/revision/{rev_id}/bare"
    resp = _shared_session.get(url, timeout=FETCH_TIMEOUT_S, headers=_headers(wiki_id))
    if resp.status_code == 404:
        raise FetchError(f"Revision {rev_id} not found on {wiki_id}", status=404)
    if not resp.ok:
        raise FetchError(
            f"Revision meta fetch failed ({resp.status_code})", status=resp.status_code
        )
    return resp.json()


def fetch_revision_html(wiki_id: str, rev_id: int) -> str:
    """Fetch Parsoid HTML for an exact revision (``/revision/{id}/html``)."""
    url = f"{rest_base(wiki_id)}/revision/{rev_id}/html"
    resp = _shared_session.get(url, timeout=FETCH_TIMEOUT_S, headers=_headers(wiki_id))
    if resp.status_code == 404:
        raise FetchError(f"Revision {rev_id} not found on {wiki_id}", status=404)
    if not resp.ok:
        raise FetchError(
            f"Revision HTML fetch failed ({resp.status_code})",
            status=resp.status_code,
        )
    return resp.text
