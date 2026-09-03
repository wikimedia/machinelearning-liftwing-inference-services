"""
Tests for the wikipedia_semantic_search tool (the P95917 port).

The requests session is patched at the module under test; no network.
"""

from unittest.mock import MagicMock, patch

import pytest

from tool_server.errors import ToolInputError, ToolUpstreamError
from tool_server.tools.wikipedia_semantic_search import (
    wikipedia_semantic_search as ws_module,
)

ws = ws_module


def _resp(ok=True, status=200, json_data=None, raise_exc=None):
    r = MagicMock()
    r.ok = ok
    r.status_code = status
    if raise_exc:
        r.raise_for_status.side_effect = raise_exc
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _search_hit(title, snippet, section=None):
    hit = {"title": title, "snippet": snippet}
    if section:
        hit["sectiontitle"] = section
    return hit


def _action_response(hits):
    return _resp(json_data={"query": {"search": hits}})


class TestWikipediaSemanticSearch:
    def test_formats_hits_with_enrichment_and_fallback(self):
        session = MagicMock()
        session.get.side_effect = [
            _action_response(
                [
                    _search_hit(
                        "Hopfield network",
                        "A <span>Hopfield</span> network &amp; memory",
                        section="Human memory",
                    ),
                    _search_hit("Catastrophic interference", "forgetting <b>x</b>"),
                ]
            ),
            # enrichment for hit 1 succeeds
            _resp(
                json_data={
                    "extract": "A Hopfield network is a form of recurrent neural network.",
                    "description": "Form of artificial neural network",
                    "content_urls": {
                        "desktop": {
                            "page": "https://en.wikipedia.org/wiki/Hopfield_network"
                        }
                    },
                }
            ),
            # enrichment for hit 2 fails -> snippet fallback
            _resp(ok=False, status=503),
        ]
        with patch.object(ws, "_session", return_value=session):
            out = ws.wikipedia_semantic_search("why do neural networks forget")

        assert out.startswith(
            "Wikipedia results for 'why do neural networks forget' (en)."
        )
        # in-band citation instruction preserved from P95917
        assert "cite" in out and "Source URL" in out
        assert "do not claim an article covers a topic" in out
        # hit 1: enriched extract, description, section, canonical url
        assert (
            "1. Hopfield network (Form of artificial neural network) [section: Human memory]"
            in out
        )
        assert "recurrent neural network." in out
        assert "Source: https://en.wikipedia.org/wiki/Hopfield_network" in out
        # hit 2: snippet fallback, html stripped, entity decoded, built url
        assert "2. Catastrophic interference" in out
        assert "forgetting x" in out
        assert "Source: https://en.wikipedia.org/wiki/Catastrophic_interference" in out
        hit2_extract = out.split("2. Catastrophic interference")[1].split("Source:")[0]
        assert "<" not in hit2_extract

    def test_semantic_parameter_and_limit_clamp(self):
        session = MagicMock()
        session.get.side_effect = [_action_response([])]
        with patch.object(ws, "_session", return_value=session):
            out = ws.wikipedia_semantic_search("q", limit=99)
        params = session.get.call_args_list[0].kwargs["params"]
        assert "cirrusSemanticSearch" in params
        assert params["srlimit"] == str(ws.MAX_LIMIT)
        assert params["srnamespace"] == "0"
        assert "No Wikipedia articles found" in out

    def test_totalhits_never_surfaced(self):
        session = MagicMock()
        session.get.side_effect = [
            _resp(
                json_data={
                    "query": {
                        "searchinfo": {"totalhits": 4321},
                        "search": [_search_hit("T", "s")],
                    }
                }
            ),
            _resp(ok=False, status=503),
        ]
        with patch.object(ws, "_session", return_value=session):
            out = ws.wikipedia_semantic_search("q")
        assert "4321" not in out

    def test_empty_query_is_deterministic_error(self):
        with pytest.raises(ToolInputError):
            ws.wikipedia_semantic_search("   ")

    def test_bad_language_falls_back_to_default(self):
        session = MagicMock()
        session.get.side_effect = [_action_response([])]
        with patch.object(ws, "_session", return_value=session):
            out = ws.wikipedia_semantic_search("q", language="EN GB!!")
        assert f"{ws.DEFAULT_LANGUAGE}.wikipedia.org" in out
        url = session.get.call_args_list[0].args[0]
        assert url.startswith(f"https://{ws.DEFAULT_LANGUAGE}.wikipedia.org")

    def test_transport_failure_is_retryable(self):
        import requests as requests_lib

        session = MagicMock()
        session.get.side_effect = requests_lib.ConnectionError("boom")
        with patch.object(ws, "_session", return_value=session):
            with pytest.raises(ToolUpstreamError):
                ws.wikipedia_semantic_search("q")

    def test_extract_truncates_on_word_boundary(self):
        long_extract = "word " * 200  # 1000 chars
        session = MagicMock()
        session.get.side_effect = [
            _action_response([_search_hit("Long", "snippet")]),
            _resp(json_data={"extract": long_extract.strip(), "content_urls": {}}),
        ]
        with patch.object(ws, "_session", return_value=session):
            out = ws.wikipedia_semantic_search("q")
        body = out.split("1. Long\n   ")[1].split("\n   Source:")[0]
        assert body.endswith("...")
        assert len(body) <= ws.EXTRACT_MAX_CHARS + 3

    def test_proxy_mode_uses_proxy_base_and_host_header(self):
        session = MagicMock()
        session.get.side_effect = [_action_response([])]
        with patch.object(ws, "MW_API_PROXY", "http://localhost:6500"):
            with patch.object(ws, "_session", return_value=session):
                ws.wikipedia_semantic_search("q", language="sw")
        call = session.get.call_args_list[0]
        assert call.args[0].startswith("http://localhost:6500/w/api.php")
        assert call.kwargs["headers"]["Host"] == "sw.wikipedia.org"

    def test_direct_mode_sends_no_host_header(self):
        session = MagicMock()
        session.get.side_effect = [_action_response([])]
        with patch.object(ws, "_session", return_value=session):
            ws.wikipedia_semantic_search("q")
        assert "Host" not in session.get.call_args_list[0].kwargs["headers"]


class TestUpstreamStatusTaxonomy:
    """Retryability per upstream status: the rule lives in one helper, so
    these cases pin it for every tool that calls MediaWiki."""

    @staticmethod
    def _fail_with(status):
        import requests as requests_lib

        resp = MagicMock()
        resp.status_code = status
        resp.url = "https://en.wikipedia.org/w/api.php"
        err = requests_lib.HTTPError(f"HTTP {status}", response=resp)
        session = MagicMock()
        search = MagicMock()
        search.raise_for_status.side_effect = err
        session.get.return_value = search
        return session

    @pytest.mark.parametrize("status", [400, 403, 404])
    def test_client_errors_are_deterministic(self, status):
        with patch.object(ws, "_session", return_value=self._fail_with(status)):
            with pytest.raises(ToolInputError) as e:
                ws.wikipedia_semantic_search("q")
        assert str(status) in str(e.value)

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_rate_limit_and_server_errors_are_retryable(self, status):
        with patch.object(ws, "_session", return_value=self._fail_with(status)):
            with pytest.raises(ToolUpstreamError):
                ws.wikipedia_semantic_search("q")

    def test_unreadable_json_is_retryable_not_swallowed(self):
        """JSONDecodeError subclasses RequestException AND ValueError, so
        this pins that it is caught by its own branch first."""
        import requests as requests_lib

        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.side_effect = requests_lib.exceptions.JSONDecodeError(
            "no json", "doc", 0
        )
        session = MagicMock()
        session.get.return_value = resp
        with patch.object(ws, "_session", return_value=session):
            with pytest.raises(ToolUpstreamError, match="unreadable response"):
                ws.wikipedia_semantic_search("q")


class TestEnrichmentLogging:
    def test_failed_enrichment_warns(self, caplog):
        session = MagicMock()
        session.get.side_effect = [
            _action_response([_search_hit("T", "snippet text")]),
            _resp(ok=False, status=503),
        ]
        with patch.object(ws, "_session", return_value=session):
            with caplog.at_level("WARNING"):
                out = ws.wikipedia_semantic_search("q")
        assert "summary enrichment failed" in caplog.text
        # the tool still succeeds on the snippet fallback
        assert "snippet text" in out


class TestStripHtml:
    def test_strips_tags_and_entities(self):
        assert (
            ws._strip_html('a <span class="x">b</span> &amp; c&nbsp;d') == "a b & c d"
        )

    def test_non_string_is_empty(self):
        assert ws._strip_html(None) == ""
