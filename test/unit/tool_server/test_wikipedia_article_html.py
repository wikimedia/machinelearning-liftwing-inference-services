"""
Tests for the wikipedia_article_html tool.

The requests session is patched at the module under test; no network.
"""

from unittest.mock import MagicMock, patch

import pytest

from tool_server.errors import ToolInputError, ToolUpstreamError
from tool_server.tools.wikipedia_article_html import (
    wikipedia_article_html as html_module,
)

wa = html_module


def _resp(text="<p>Jupiter</p>", status=200, raise_exc=None):
    r = MagicMock()
    r.status_code = status
    r.text = text
    if raise_exc:
        r.raise_for_status.side_effect = raise_exc
    return r


class TestWikipediaArticleHtml:
    def test_returns_html_body(self):
        session = MagicMock()
        session.get.return_value = _resp("<html><body>Jupiter</body></html>")
        with patch.object(wa, "_session", return_value=session):
            out = wa.wikipedia_article_html("Jupiter")
        assert out == "<html><body>Jupiter</body></html>"
        call = session.get.call_args
        assert (
            call.args[0] == "https://en.wikipedia.org/w/rest.php/v1/page/Jupiter/html"
        )
        assert call.kwargs["params"] == {"flavor": "view"}
        assert call.kwargs["headers"]["Accept"] == "text/html"
        assert "Host" not in call.kwargs["headers"]

    def test_quotes_title_as_one_path_segment(self):
        session = MagicMock()
        session.get.return_value = _resp("<p>ok</p>")
        with patch.object(wa, "_session", return_value=session):
            wa.wikipedia_article_html("A/B test")
        url = session.get.call_args.args[0]
        assert url.endswith("/w/rest.php/v1/page/A%2FB_test/html")

    def test_under_cap_returns_html_alone(self):
        body = "x" * wa.HTML_MAX_CHARS
        session = MagicMock()
        session.get.return_value = _resp(body)
        with patch.object(wa, "_session", return_value=session):
            out = wa.wikipedia_article_html("Jupiter")
        assert out == body
        assert "truncated" not in out

    def test_truncates_and_prefixes_source_note(self):
        body = "y" * (wa.HTML_MAX_CHARS + 50)
        session = MagicMock()
        session.get.return_value = _resp(body)
        with patch.object(wa, "_session", return_value=session):
            out = wa.wikipedia_article_html("John von Neumann")
        note, html = out.split("\n", 1)
        assert note == (
            f"[HTML truncated to {wa.HTML_MAX_CHARS} characters. "
            "Source: https://en.wikipedia.org/wiki/John_von_Neumann]"
        )
        assert html == "y" * wa.HTML_MAX_CHARS

    def test_empty_title_is_deterministic_error(self):
        with pytest.raises(ToolInputError):
            wa.wikipedia_article_html("   ")

    def test_bad_language_falls_back_to_default(self):
        session = MagicMock()
        session.get.return_value = _resp("<p>x</p>")
        with patch.object(wa, "_session", return_value=session):
            wa.wikipedia_article_html("Jupiter", language="EN GB!!")
        url = session.get.call_args.args[0]
        assert url.startswith(f"https://{wa.DEFAULT_LANGUAGE}.wikipedia.org")

    def test_proxy_mode_uses_proxy_base_and_host_header(self):
        session = MagicMock()
        session.get.return_value = _resp("<p>x</p>")
        with patch.object(wa, "MW_API_PROXY", "http://localhost:6500"):
            with patch.object(wa, "_session", return_value=session):
                wa.wikipedia_article_html("Jupiter", language="sw")
        call = session.get.call_args
        assert call.args[0].startswith(
            "http://localhost:6500/w/rest.php/v1/page/Jupiter/html"
        )
        assert call.kwargs["headers"]["Host"] == "sw.wikipedia.org"
        assert call.kwargs["headers"]["Accept"] == "text/html"

    def test_direct_mode_sends_no_host_header(self):
        session = MagicMock()
        session.get.return_value = _resp("<p>x</p>")
        with patch.object(wa, "_session", return_value=session):
            wa.wikipedia_article_html("Jupiter")
        assert "Host" not in session.get.call_args.kwargs["headers"]

    def test_transport_failure_is_retryable(self):
        import requests as requests_lib

        session = MagicMock()
        session.get.side_effect = requests_lib.ConnectionError("boom")
        with patch.object(wa, "_session", return_value=session):
            with pytest.raises(ToolUpstreamError):
                wa.wikipedia_article_html("Jupiter")

    def test_unreadable_body_is_retryable(self):
        session = MagicMock()
        session.get.return_value = _resp(text=None)
        with patch.object(wa, "_session", return_value=session):
            with pytest.raises(ToolUpstreamError, match="unreadable body"):
                wa.wikipedia_article_html("Jupiter")


class TestUpstreamStatusTaxonomy:
    @staticmethod
    def _fail_with(status):
        import requests as requests_lib

        resp = MagicMock()
        resp.status_code = status
        resp.url = "https://en.wikipedia.org/w/rest.php/v1/page/Missing/html"
        err = requests_lib.HTTPError(f"HTTP {status}", response=resp)
        session = MagicMock()
        page = MagicMock()
        page.raise_for_status.side_effect = err
        session.get.return_value = page
        return session

    @pytest.mark.parametrize("status", [400, 403, 404])
    def test_client_errors_are_deterministic(self, status):
        with patch.object(wa, "_session", return_value=self._fail_with(status)):
            with pytest.raises(ToolInputError) as e:
                wa.wikipedia_article_html("Missing")
        assert str(status) in str(e.value)

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_rate_limit_and_server_errors_are_retryable(self, status):
        with patch.object(wa, "_session", return_value=self._fail_with(status)):
            with pytest.raises(ToolUpstreamError):
                wa.wikipedia_article_html("Jupiter")
