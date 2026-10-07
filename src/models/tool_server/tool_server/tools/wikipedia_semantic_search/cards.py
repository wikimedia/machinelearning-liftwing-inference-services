"""
The MCP App version of wikipedia_semantic_search: results as a card grid.

MCP Apps hosts (e.g. LibreChat, Claude) render the View in ``ui/grid.html``
with the structured payload built here; the model still reads the same text
as from the plain tool. Other MCP clients ignore the UI metadata and get
that text. REST is unaffected.

The View comes from the Wikimedia hackathon project
https://gitlab.wikimedia.org/egardner/wikipedia-semantic-search-mcp; see
``ui/README.md`` for the exact build. The payload below is the
``SearchPayload`` shape it expects (``ui/types.ts`` there).
"""

from pathlib import Path
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from tool_server.config import DEFAULT_LANGUAGE, DEFAULT_LIMIT, MAX_LIMIT
from tool_server.errors import ToolInputError, ToolUpstreamError
from tool_server.registry import app_tool
from tool_server.tools.wikipedia_semantic_search.search import SearchResults, search
from tool_server.tools.wikipedia_semantic_search.wikipedia_semantic_search import (
    format_text,
)

RESOURCE_URI = "ui://wikipedia-semantic-search/grid.html"

# Where the View may load images from: lead images (the page summary links
# thumb.wikimedia.org as of 2026-10; upload.wikimedia.org earlier) and
# wiki-hosted assets. Anything else is blocked and shows the placeholder.
RESOURCE_DOMAINS = (
    "https://thumb.wikimedia.org",
    "https://upload.wikimedia.org",
    "https://*.wikipedia.org",
)


@app_tool(
    tool="wikipedia_semantic_search",
    resource_uri=RESOURCE_URI,
    html=Path(__file__).parent / "ui" / "grid.html",
    resource_domains=RESOURCE_DOMAINS,
)
def wikipedia_semantic_search_cards(
    query: Annotated[str, Field(description="Natural-language search query.")],
    language: Annotated[
        str, Field(description="Wikipedia language code, e.g. 'en', 'de', 'sw'.")
    ] = DEFAULT_LANGUAGE,
    limit: Annotated[
        int, Field(ge=1, le=MAX_LIMIT, description=f"Max results (1-{MAX_LIMIT}).")
    ] = DEFAULT_LIMIT,
) -> ToolResult:
    try:
        results = search(query, language, limit)
    # MCP has no status codes: both kinds become a tool error the model
    # can read, as the REST-derived tools' 400 and 502 do.
    except (ToolInputError, ToolUpstreamError) as e:
        raise ToolError(str(e)) from e
    return ToolResult(
        content=format_text(results), structured_content=to_payload(results)
    )


def to_payload(results: SearchResults) -> dict:
    """The View's ``SearchPayload``. Fields we do not fetch yet are null."""
    return {
        "query": results.query,
        "language": results.language,
        # Not surfaced, for the reason given in search.search.
        "totalHits": None,
        "results": [
            {
                "title": hit.title,
                "pageId": hit.page_id,
                "snippet": hit.snippet,
                "section": hit.section,
                "extract": hit.extract,
                "description": hit.description,
                "image": (
                    {
                        "src": hit.thumbnail.src,
                        "srcset": None,
                        "lqip": None,
                        "width": hit.thumbnail.width,
                        "height": hit.thumbnail.height,
                    }
                    if hit.thumbnail
                    else None
                ),
                "url": hit.url,
                # The View opens this on click. The hackathon server links
                # to the matching passage (#:~:text=); the article will do
                # until that is ported.
                "textFragmentUrl": hit.url,
                # Attribution API signals: a future enrichment in search.
                "attribution": None,
            }
            for hit in results.hits
        ],
    }
