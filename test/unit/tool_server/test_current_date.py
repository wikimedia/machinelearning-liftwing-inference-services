"""Tests for the current_date tool."""

import pytest

from tool_server.errors import ToolInputError
from tool_server.tools.current_date.current_date import current_date


class TestCurrentDate:
    def test_known_timezone_fields(self):
        out = current_date("Africa/Kampala")
        assert set(out) == {"iso", "date", "day", "timezone"}
        assert out["timezone"] == "Africa/Kampala"
        assert out["iso"].endswith("+03:00")
        assert out["iso"].startswith(out["date"])

    def test_default_is_utc(self):
        out = current_date()
        assert out["timezone"] == "UTC"
        assert out["iso"].endswith("+00:00")

    def test_unknown_timezone_is_deterministic_error(self):
        with pytest.raises(ToolInputError, match="Unknown timezone"):
            current_date("Mars/Olympus_Mons")
