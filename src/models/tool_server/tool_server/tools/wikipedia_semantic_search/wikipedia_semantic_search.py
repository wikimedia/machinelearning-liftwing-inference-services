import logging
import re
from typing import Annotated

import requests
from fastapi import Query
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
from tool_server.registry import tool

logger = logging.getLogger(__name__)

_LANG_OK = re.compile(r"^[a-z0-9-]{2,12}$")
# 429 is rate limiting: the same request succeeds once the window clears,
# so it is transient even though it is a 4xx.
_RETRYABLE_4XX = frozenset({429})


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


@tool
def wikipedia_semantic_search(
    query: Annotated[str, Query(description="Natural-language search query.")],
    language: Annotated[
        str,
        Query(description="Wikipedia language code, e.g. 'en', 'de', 'sw'."),
    ] = DEFAULT_LANGUAGE,
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description=f"Max results (1-{MAX_LIMIT}).")
    ] = DEFAULT_LIMIT,
) -> str:
    """
    Search Wikipedia for articles matching a natural-language query.

    Uses semantic (meaning-based) search where available, falling back to
    keyword search. Returns titles, descriptions, extracts, and source
    URLs. Cite the Source URL for each article you use.
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
        hits = r.json().get("query", {}).get("search", [])
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

    if not hits:
        return f"No Wikipedia articles found for '{query}' on {language}.wikipedia.org."

    # 2. Per-hit enrichment via the REST page-summary endpoint (extract,
    #    short description, canonical URL). A failed summary call degrades
    #    gracefully to the search snippet, which also keeps the tool
    #    functional if the services-proxy does not route /api/rest_v1
    #    (to be verified at staging deploy).
    lines = []
    for i, hit in enumerate(hits, 1):
        title = hit.get("title", "")
        if not title:
            continue
        snippet = _strip_html(hit.get("snippet", ""))
        section = hit.get("sectiontitle")
        extract = snippet
        description = None
        url = (
            f"https://{language}.wikipedia.org/wiki/"
            f"{requests.utils.quote(title.replace(' ', '_'))}"
        )

        try:
            s = session.get(
                f"{base}/api/rest_v1/page/summary/"
                f"{requests.utils.quote(title, safe='')}",
                headers=headers,
                timeout=HTTP_TIMEOUT_S,
            )
            if s.ok:
                data = s.json()
                extract = data.get("extract") or snippet
                description = data.get("description")
                url = data.get("content_urls", {}).get("desktop", {}).get("page") or url
            else:
                # A non-ok response is not an exception, so it needs its own
                # log line: otherwise degraded enrichment is invisible.
                logger.warning(
                    "summary enrichment failed for %r: HTTP %s",
                    title,
                    s.status_code,
                )
        except (requests.RequestException, ValueError) as e:
            # Not fatal (the snippet is the fallback), but it means the
            # summary endpoint or its route is unhealthy, so it is visible.
            logger.warning("summary enrichment failed for %r: %s", title, e)

        if len(extract) > EXTRACT_MAX_CHARS:
            extract = extract[:EXTRACT_MAX_CHARS].rsplit(" ", 1)[0] + "..."

        desc = f" ({description})" if description else ""
        sect = f" [section: {section}]" if section else ""
        lines.append(f"{i}. {title}{desc}{sect}\n   {extract}\n   Source: {url}")

    if not lines:
        return f"No usable Wikipedia results for '{query}' on {language}.wikipedia.org."

    # In-band instruction. Asking for citations in the tool output rather
    # than relying on the user's phrasing makes the behaviour hold on every
    # call. The second clause targets residual overclaiming: on 2026-08-05
    # a 14B model summarised unrelated articles as though they answered the
    # query, and requesting citations reduced but did not remove this.
    return (
        f"Wikipedia results for '{query}' ({language}). When answering, cite "
        "the Source URL for each article you reference, and do not claim an "
        "article covers a topic unless its extract says so:\n\n" + "\n\n".join(lines)
    )


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
