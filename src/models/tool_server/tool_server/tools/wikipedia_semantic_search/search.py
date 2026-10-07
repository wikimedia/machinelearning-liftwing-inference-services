"""
The Wikipedia search behind wikipedia_semantic_search.

Runs the search, fetches each page's summary, and returns the results as
data. wikipedia_semantic_search.py turns them into text for the model, and
cards.py into the data for the card grid.
"""

import logging
import re
from dataclasses import dataclass

import requests
from urllib3.util.retry import Retry

from tool_server.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_LIMIT,
    EXTRACT_MAX_CHARS,
    HTTP_RETRIES,
    HTTP_TIMEOUT_S,
    MAX_LIMIT,
    MW_API_PROXY,
    USER_AGENT,
)
from tool_server.errors import ToolInputError, ToolUpstreamError

logger = logging.getLogger(__name__)

_LANG_OK = re.compile(r"^[a-z0-9-]{2,12}$")
# 429 is rate limiting: the same request succeeds once the window clears,
# so it is transient even though it is a 4xx.
_RETRYABLE_4XX = frozenset({429})


@dataclass(frozen=True)
class Thumbnail:
    """The article's lead image, as the page summary reports it."""

    src: str
    width: int | None
    height: int | None


@dataclass(frozen=True)
class Hit:
    """One search result after enrichment."""

    title: str
    page_id: int | None
    snippet: str
    section: str | None
    # The summary extract, or the snippet when enrichment failed. Already
    # truncated to EXTRACT_MAX_CHARS.
    extract: str
    description: str | None
    url: str
    thumbnail: Thumbnail | None


@dataclass(frozen=True)
class SearchResults:
    """The normalized request and what it found."""

    query: str
    language: str
    hits: list[Hit]
    # How many results the search returned before hits without a title
    # were dropped, so a presenter can tell "nothing found" from "nothing
    # usable".
    found: int


def search(query: str, language: str, limit: int) -> SearchResults:
    """
    Search one Wikipedia and enrich each hit with its page summary.

    Normalizes the request first: an empty query is a ToolInputError, an
    invalid language falls back to the default, and limit is clamped to
    1..MAX_LIMIT.
    """
    query = (query or "").strip()
    if not query:
        raise ToolInputError("No search query provided.")
    if not isinstance(language, str) or not _LANG_OK.match(language):
        language = DEFAULT_LANGUAGE
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    base = _wiki_base(language)
    headers = _wiki_headers(language)
    session = _session()

    # 1. Action API list=search with the experimental semantic parameter.
    #    The action API's parameter validator does not declare
    #    cirrusSemanticSearch and so returns an "Unrecognized parameter"
    #    warning, but the search backend still honours it: verified on
    #    en.wikipedia.org on 2026-08-05 by comparing result sets with and
    #    without the parameter (deterministic and clearly different).
    #    On wikis where it has no effect, results degrade to keyword
    #    ranking, so callers always get a result list.
    #    searchinfo.totalhits is deliberately not surfaced: under
    #    semantic search it tracks the candidate-set size (it scales with
    #    srlimit) rather than a real match count, so reporting it would
    #    invite the model to state a fabricated total.
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "list": "search",
        "srsearch": query,
        "srprop": "snippet|sectiontitle",
        "srlimit": str(limit),
        "srnamespace": "0",
        "cirrusSemanticSearch": "",
    }
    try:
        r = session.get(
            f"{base}/w/api.php", params=params, headers=headers, timeout=HTTP_TIMEOUT_S
        )
        r.raise_for_status()
        raw_hits = r.json().get("query", {}).get("search", [])
    # Order matters: requests.exceptions.JSONDecodeError subclasses both
    # RequestException and ValueError, so it must be caught first or a
    # later branch swallows it.
    except requests.exceptions.JSONDecodeError as e:
        raise ToolUpstreamError(
            f"Wikipedia search returned an unreadable response: {e}"
        ) from e
    except requests.HTTPError as e:
        _raise_for_upstream_status(e, "Wikipedia search")
    except requests.RequestException as e:
        raise ToolUpstreamError(f"Wikipedia search request failed: {e}") from e

    hits = []
    for raw in raw_hits:
        if raw.get("title"):
            hits.append(_enrich(raw, language, base, headers, session))
    return SearchResults(query=query, language=language, hits=hits, found=len(raw_hits))


def _enrich(
    raw: dict, language: str, base: str, headers: dict, session: requests.Session
) -> Hit:
    """
    Per-hit enrichment via the REST page-summary endpoint (extract, short
    description, canonical URL, lead image). A failed summary call
    degrades gracefully to the search snippet, which also keeps the tool
    functional if the services-proxy does not route /api/rest_v1 (to be
    verified at staging deploy).
    """
    title = raw["title"]
    snippet = _strip_html(raw.get("snippet", ""))
    extract = snippet
    description = None
    thumbnail = None
    url = (
        f"https://{language}.wikipedia.org/wiki/"
        f"{requests.utils.quote(title.replace(' ', '_'))}"
    )

    try:
        s = session.get(
            f"{base}/api/rest_v1/page/summary/{requests.utils.quote(title, safe='')}",
            headers=headers,
            timeout=HTTP_TIMEOUT_S,
        )
        if s.ok:
            data = s.json()
            extract = data.get("extract") or snippet
            description = data.get("description")
            url = data.get("content_urls", {}).get("desktop", {}).get("page") or url
            thumbnail = _thumbnail(data.get("thumbnail"))
        else:
            # A non-ok response is not an exception, so it needs its own
            # log line: otherwise degraded enrichment is invisible.
            logger.warning(
                "summary enrichment failed for %r: HTTP %s", title, s.status_code
            )
    except (requests.RequestException, ValueError) as e:
        # Not fatal (the snippet is the fallback), but it means the
        # summary endpoint or its route is unhealthy, so it is visible.
        logger.warning("summary enrichment failed for %r: %s", title, e)

    if len(extract) > EXTRACT_MAX_CHARS:
        extract = extract[:EXTRACT_MAX_CHARS].rsplit(" ", 1)[0] + "..."

    return Hit(
        title=title,
        page_id=raw.get("pageid"),
        snippet=snippet,
        section=raw.get("sectiontitle"),
        extract=extract,
        description=description,
        url=url,
        thumbnail=thumbnail,
    )


def _thumbnail(raw: object) -> Thumbnail | None:
    """The summary's ``thumbnail`` object, or None when it has no source."""
    if not isinstance(raw, dict) or not raw.get("source"):
        return None
    return Thumbnail(
        src=raw["source"], width=raw.get("width"), height=raw.get("height")
    )


def _raise_for_upstream_status(e: requests.HTTPError, what: str) -> None:
    """
    Map an upstream HTTP error onto the taxonomy.

    A 4xx from MediaWiki means our request was wrong for this wiki and
    will fail identically next time, so it is deterministic (400). A 5xx,
    or a 429, is transient (502).
    """
    status = getattr(e.response, "status_code", None)
    if status is not None and 400 <= status < 500 and status not in _RETRYABLE_4XX:
        raise ToolInputError(
            f"{what} was rejected by {getattr(e.response, 'url', 'Wikipedia')!s} "
            f"with HTTP {status}."
        ) from e
    raise ToolUpstreamError(f"{what} failed with HTTP {status}.") from e


def _wiki_base(language: str) -> str:
    """
    Base URL for one language edition.

    When TOOL_MW_API_PROXY is set, the URL points at the envoy
    services-proxy (e.g. ``http://localhost:6500``) instead of hitting
    Wikipedia directly, and requests must also carry the Host header
    from :func:`_wiki_headers`.
    """
    if MW_API_PROXY:
        return MW_API_PROXY
    return f"https://{language}.wikipedia.org"


def _wiki_headers(language: str) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if MW_API_PROXY:
        # LiftWing's services-proxy routes on the Host header.
        headers["Host"] = f"{language}.wikipedia.org"
    return headers


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
    s.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    return s


def _strip_html(html: str) -> str:
    """Strip the lightweight HTML the action API includes in snippets."""
    if not isinstance(html, str):
        return ""
    text = re.sub(r"<[^>]+>", "", html)
    for a, b in [
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ]:
        text = text.replace(a, b)
    return " ".join(text.split())
