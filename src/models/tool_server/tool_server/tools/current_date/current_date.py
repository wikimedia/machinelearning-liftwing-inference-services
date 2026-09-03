from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Query

from tool_server.errors import ToolInputError
from tool_server.registry import tool


@tool
def current_date(
    timezone: Annotated[
        str,
        Query(description='IANA timezone name, e.g. "UTC" or "Africa/Kampala".'),
    ] = "UTC",
) -> dict:
    """
    Get the current date and time in a timezone.

    Use for any question about today's date, the current day of the week,
    or the time right now.

    An unknown timezone is a deterministic failure: the same request
    will always fail the same way, so it raises ToolInputError (400).
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ToolInputError(
            f"Unknown timezone {timezone!r}. Use an IANA name such as "
            '"UTC", "Africa/Kampala" or "Europe/Berlin".'
        ) from e
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
        "timezone": timezone,
    }
