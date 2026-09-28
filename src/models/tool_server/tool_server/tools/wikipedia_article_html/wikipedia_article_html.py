import logging
import re
from typing import Annotated

import requests
from fastapi import Query
from urllib3.util.retry import Retry

from tool_server.config import (
    DEFAULT_LANGUAGE,
    HTML_MAX_CHARS,
    HTTP_RETRIES,
    HTTP_TIMEOUT_S,
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
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
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


def _quote_title(title: str) -> str:
    """Encode a title as a single REST path segment."""
    return requests.utils.quote(title.replace(" ", "_"), safe="")


def _article_url(language: str, title: str) -> str:
    return f"https://{language}.wikipedia.org/wiki/{_quote_title(title)}"


@tool
def wikipedia_article_html(
    title: Annotated[
        str, Query(description="Wikipedia article title, e.g. 'Jupiter'.")
    ],
    language: Annotated[
        str,
        Query(description="Wikipedia language code, e.g. 'en', 'de', 'sw'."),
    ] = DEFAULT_LANGUAGE,
) -> str:
    """
    Fetch the HTML of a Wikipedia article by title.

    Returns the latest page content as HTML from the MediaWiki REST API.
    Long pages are truncated; the response says so and includes the
    source URL when that happens.
    """
    title = (title or "").strip()
    if not title:
        raise ToolInputError("No article title provided.")
    if not isinstance(language, str) or not _LANG_OK.match(language):
        language = DEFAULT_LANGUAGE

    base = _wiki_base(language)
    headers = _wiki_headers(language)
    url = f"{base}/w/rest.php/v1/page/{_quote_title(title)}/html"
    try:
        r = _session().get(
            url,
            params={"flavor": "view"},
            headers=headers,
            timeout=HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        html = r.text
    except requests.HTTPError as e:
        _raise_for_upstream_status(e, "Wikipedia article HTML")
    except requests.RequestException as e:
        raise ToolUpstreamError(f"Wikipedia article HTML request failed: {e}") from e

    if not isinstance(html, str):
        raise ToolUpstreamError("Wikipedia article HTML returned an unreadable body.")

    if len(html) > HTML_MAX_CHARS:
        source = _article_url(language, title)
        logger.info(
            "truncated HTML for %r (%s) from %d to %d characters",
            title,
            language,
            len(html),
            HTML_MAX_CHARS,
        )
        note = f"[HTML truncated to {HTML_MAX_CHARS} characters. Source: {source}]\n"
        return note + html[:HTML_MAX_CHARS]
    return html
