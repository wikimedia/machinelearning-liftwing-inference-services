"""
Service tests: both protocols from one codebase.

REST via FastAPI's TestClient; MCP via fastmcp's in-memory client
against the derived server, which pins that route exclusion and tool
naming survive refactors.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client

from tool_server import config, service


@pytest.fixture(scope="module")
def client():
    """
    One TestClient (and so one lifespan) for the whole module: the MCP
    session manager mounted at /mcp can only be started once per process.
    """
    with TestClient(service.app) as c:
        yield c


class TestRest:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_current_date_route(self, client):
        r = client.get("/v1/tools/current-date", params={"timezone": "Africa/Kampala"})
        assert r.status_code == 200
        assert r.json()["day"] in {
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        }

    def test_unknown_timezone_is_400(self, client):
        r = client.get("/v1/tools/current-date", params={"timezone": "Nope/Nowhere"})
        assert r.status_code == 400
        assert "Unknown timezone" in r.json()["detail"]

    def test_input_error_maps_to_400(self, client):
        """Through the real tool: an unknown timezone is deterministic."""
        r = client.get("/v1/tools/current-date", params={"timezone": "Nope/Nowhere"})
        assert r.status_code == 400
        assert "Unknown timezone" in r.json()["detail"]

    def test_upstream_error_maps_to_502(self, client):
        """Through the real tool: a transport failure is retryable."""
        import requests

        from tool_server.tools.wikipedia_semantic_search import (
            wikipedia_semantic_search as ws,
        )

        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("boom")
        with patch.object(ws, "_session", return_value=session):
            r = client.get("/v1/tools/wikipedia-semantic-search", params={"query": "x"})
        assert r.status_code == 502
        assert "failed" in r.json()["detail"]

    def test_search_route_passes_parameters_through(self, client):
        """The route's signature is the tool's signature, so parameters
        arrive as declared."""
        from tool_server.tools.wikipedia_semantic_search import (
            wikipedia_semantic_search as ws,
        )

        session = MagicMock()
        session.get.return_value = MagicMock(
            ok=True,
            **{
                "raise_for_status.return_value": None,
                "json.return_value": {"query": {"search": []}},
            },
        )
        with patch.object(ws, "_session", return_value=session):
            r = client.get(
                "/v1/tools/wikipedia-semantic-search",
                params={"query": "CRISPR", "language": "de", "limit": 3},
            )
        assert r.status_code == 200
        params = session.get.call_args.kwargs["params"]
        assert params["srsearch"] == "CRISPR"
        assert params["srlimit"] == "3"
        assert "de.wikipedia.org" in r.json()

    def test_openapi_contract_names_the_tools(self, client):
        spec = client.get("/openapi.json").json()
        ids = {
            op["operationId"]
            for methods in spec["paths"].values()
            for op in methods.values()
        }
        assert {"current_date", "wikipedia_semantic_search"} <= ids
        assert "/healthz" not in spec["paths"]

    def test_search_limits_come_from_config(self, client):
        """The published contract must be generated from config, so a
        changed limit cannot disagree with what the tool enforces."""
        spec = client.get("/openapi.json").json()
        params = {
            p["name"]: p
            for p in spec["paths"]["/v1/tools/wikipedia-semantic-search"]["get"][
                "parameters"
            ]
        }
        assert params["limit"]["schema"]["maximum"] == config.MAX_LIMIT
        assert params["limit"]["schema"]["default"] == config.DEFAULT_LIMIT
        assert params["language"]["schema"]["default"] == config.DEFAULT_LANGUAGE

    def test_limit_above_max_is_rejected_by_the_contract(self, client):
        r = client.get(
            "/v1/tools/wikipedia-semantic-search",
            params={"query": "x", "limit": config.MAX_LIMIT + 1},
        )
        assert r.status_code == 422


class TestMcp:
    def test_tool_list_is_exactly_the_toolbelt(self):
        async def main():
            async with Client(service.mcp) as c:
                tools = await c.list_tools()
                return sorted(t.name for t in tools)

        assert asyncio.run(main()) == ["current_date", "wikipedia_semantic_search"]

    def test_tools_call_current_date(self):
        async def main():
            async with Client(service.mcp) as c:
                r = await c.call_tool("current_date", {"timezone": "Africa/Kampala"})
                return r.data if getattr(r, "data", None) else r.content[0].text

        out = asyncio.run(main())
        assert "Africa/Kampala" in str(out)
