"""
Tests for the MCP App version of wikipedia_semantic_search (cards.py).

MCP Apps hosts render the View from three things this server publishes:
the tool's ``_meta.ui.resourceUri``, the ``ui://`` resource with its CSP,
and the tool result's ``structuredContent``. These tests pin all three,
and that the model-facing text is the plain tool's text. No network.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from tool_server import service
from tool_server.tools.wikipedia_semantic_search import cards, search as search_mod
from tool_server.tools.wikipedia_semantic_search.search import (
    Hit,
    SearchResults,
    Thumbnail,
)
from tool_server.tools.wikipedia_semantic_search.wikipedia_semantic_search import (
    format_text,
)


def _resp(ok=True, status=200, json_data=None):
    r = MagicMock()
    r.ok = ok
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _session_for_one_hit():
    session = MagicMock()
    session.get.side_effect = [
        _resp(
            json_data={
                "query": {
                    "search": [
                        {
                            "title": "Hekla",
                            "pageid": 113726,
                            "snippet": "a <b>volcano</b>",
                        }
                    ]
                }
            }
        ),
        _resp(
            json_data={
                "extract": "Hekla is an active stratovolcano in southern Iceland.",
                "description": "Volcano in Iceland",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Hekla"}
                },
                "thumbnail": {
                    "source": "https://thumb.wikimedia.org/x/330px-Hekla.jpg",
                    "width": 330,
                    "height": 248,
                },
            }
        ),
    ]
    return session


def _hit(**overrides):
    fields = {
        "title": "Hekla",
        "page_id": 113726,
        "snippet": "a volcano",
        "section": None,
        "extract": "Hekla is a volcano.",
        "description": "Volcano in Iceland",
        "url": "https://en.wikipedia.org/wiki/Hekla",
        "thumbnail": Thumbnail(
            src="https://thumb.wikimedia.org/h.jpg", width=330, height=248
        ),
    }
    fields.update(overrides)
    return Hit(**fields)


def _call(coro_fn):
    async def main():
        async with Client(service.mcp) as c:
            return await coro_fn(c)

    return asyncio.run(main())


class TestPayload:
    def test_maps_hits_to_the_view_payload(self):
        results = SearchResults(query="q", language="en", hits=[_hit()], found=1)
        payload = cards.to_payload(results)

        assert payload["query"] == "q"
        assert payload["language"] == "en"
        assert payload["totalHits"] is None
        (card,) = payload["results"]
        assert card["title"] == "Hekla"
        assert card["pageId"] == 113726
        assert card["description"] == "Volcano in Iceland"
        assert card["image"] == {
            "src": "https://thumb.wikimedia.org/h.jpg",
            "srcset": None,
            "lqip": None,
            "width": 330,
            "height": 248,
        }
        assert card["textFragmentUrl"] == card["url"]
        assert card["attribution"] is None

    def test_no_thumbnail_is_a_null_image(self):
        results = SearchResults(
            query="q", language="en", hits=[_hit(thumbnail=None)], found=1
        )
        assert cards.to_payload(results)["results"][0]["image"] is None


class TestSearchKeepsThumbnail:
    def test_summary_thumbnail_is_kept(self):
        with patch.object(search_mod, "_session", return_value=_session_for_one_hit()):
            results = search_mod.search("volcano", "en", 1)
        assert results.hits[0].thumbnail == Thumbnail(
            src="https://thumb.wikimedia.org/x/330px-Hekla.jpg", width=330, height=248
        )
        assert results.hits[0].page_id == 113726

    def test_failed_summary_has_no_thumbnail(self):
        session = MagicMock()
        session.get.side_effect = [
            _resp(json_data={"query": {"search": [{"title": "T", "snippet": "s"}]}}),
            _resp(ok=False, status=503),
        ]
        with patch.object(search_mod, "_session", return_value=session):
            results = search_mod.search("q", "en", 1)
        assert results.hits[0].thumbnail is None


class TestMcpApp:
    def test_tool_points_at_the_view(self):
        tools = _call(lambda c: c.list_tools())
        (tool,) = [t for t in tools if t.name == "wikipedia_semantic_search"]
        assert tool.meta["ui"]["resourceUri"] == cards.RESOURCE_URI

    def test_tool_keeps_the_plain_tool_description(self):
        tools = _call(lambda c: c.list_tools())
        (tool,) = [t for t in tools if t.name == "wikipedia_semantic_search"]
        plain = service._TOOLS_BY_NAME["wikipedia_semantic_search"]
        assert tool.description == plain.description

    def test_view_is_served_with_its_csp_on_the_read_item(self):
        """LibreChat reads the CSP from the resources/read item, not the
        listing, so it must be there."""
        (item,) = _call(lambda c: c.read_resource(cards.RESOURCE_URI))
        assert item.mime_type == "text/html;profile=mcp-app"
        assert item.text.lstrip().lower().startswith("<!doctype html>")
        csp = item.meta["ui"]["csp"]
        assert set(csp["resourceDomains"]) == set(cards.RESOURCE_DOMAINS)
        assert "https://thumb.wikimedia.org" in csp["resourceDomains"]

    def test_call_returns_plain_text_and_structured_payload(self):
        with patch.object(search_mod, "_session", return_value=_session_for_one_hit()):
            result = _call(
                lambda c: c.call_tool("wikipedia_semantic_search", {"query": "volcano"})
            )
        assert result.structured_content["results"][0]["title"] == "Hekla"
        assert result.structured_content["results"][0]["image"]["src"].startswith(
            "https://thumb.wikimedia.org/"
        )

        with patch.object(search_mod, "_session", return_value=_session_for_one_hit()):
            expected_text = format_text(search_mod.search("volcano", "en", 5))
        assert result.content[0].text == expected_text

    def test_input_error_is_a_tool_error(self):
        with pytest.raises(ToolError, match="No search query"):
            _call(lambda c: c.call_tool("wikipedia_semantic_search", {"query": "  "}))

    def test_rest_route_still_returns_plain_text(self):
        from fastapi.testclient import TestClient

        with patch.object(search_mod, "_session", return_value=_session_for_one_hit()):
            # No lifespan: REST routes do not need the MCP session manager.
            r = TestClient(service.app).get(
                "/v1/tools/wikipedia-semantic-search", params={"query": "volcano"}
            )
        assert r.status_code == 200
        assert r.json().startswith("Wikipedia results for 'volcano' (en).")
