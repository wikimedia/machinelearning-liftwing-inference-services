from typing import Annotated

from fastapi import Query

from tool_server.config import DEFAULT_LANGUAGE, DEFAULT_LIMIT, MAX_LIMIT
from tool_server.registry import tool
from tool_server.tools.wikipedia_semantic_search.search import SearchResults, search


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
    return format_text(search(query, language, limit))


def format_text(results: SearchResults) -> str:
    """The model-facing text for a search."""
    query, language = results.query, results.language
    if not results.found:
        return f"No Wikipedia articles found for '{query}' on {language}.wikipedia.org."
    if not results.hits:
        return f"No usable Wikipedia results for '{query}' on {language}.wikipedia.org."

    lines = []
    for i, hit in enumerate(results.hits, 1):
        desc = f" ({hit.description})" if hit.description else ""
        sect = f" [section: {hit.section}]" if hit.section else ""
        lines.append(
            f"{i}. {hit.title}{desc}{sect}\n   {hit.extract}\n   Source: {hit.url}"
        )

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
