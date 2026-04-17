"""
DateTimeTool — comprehensive date/time utility for conversational AI.

Supports:
- Current time in any timezone
- Time conversion between timezones
- Date calculations (add/subtract time)
- Date difference queries
"""
from __future__ import annotations

import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import Tool

# ---------------------------------------------------------------------------
# Detect the server's local IANA timezone name
# ---------------------------------------------------------------------------

def _detect_local_iana_tz() -> str:
    # 1. TZ environment variable
    tz = os.environ.get("TZ", "").strip()
    if tz:
        return tz
    # 2. /etc/timezone (Debian/Ubuntu)
    try:
        return pathlib.Path("/etc/timezone").read_text().strip()
    except FileNotFoundError:
        pass
    # 3. /etc/localtime symlink (macOS + RHEL/CentOS)
    try:
        target = pathlib.Path("/etc/localtime").resolve()
        s = str(target)
        for marker in ("zoneinfo/", "zoneinfo\\"):
            idx = s.find(marker)
            if idx != -1:
                return s[idx + len(marker):]
    except Exception:
        pass
    return "UTC"


SERVER_TIMEZONE = _detect_local_iana_tz()

# Common natural-language timezone aliases → IANA names
TIMEZONE_ALIASES: dict[str, str] = {
    # US
    "pst": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "mst": "America/Denver",
    "mountain": "America/Denver",
    "cst": "America/Chicago",
    "central": "America/Chicago",
    "est": "America/New_York",
    "eastern": "America/New_York",
    "edt": "America/New_York",
    # International
    "gmt": "Europe/London",
    "utc": "UTC",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "jst": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "kst": "Asia/Seoul",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "aest": "Australia/Sydney",
    "mumbai": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
    "dubai": "Asia/Dubai",
    "moscow": "Europe/Moscow",
    "sao paulo": "America/Sao_Paulo",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "denver": "America/Denver",
}


def _resolve_timezone(tz: str | None) -> str:
    if not tz:
        return SERVER_TIMEZONE
    normalized = tz.lower().strip()
    return TIMEZONE_ALIASES.get(normalized, tz)


def _get_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError):
        raise ValueError(f"Unknown timezone: {tz_name!r}")


def _format_datetime(dt: datetime, tz_name: str) -> dict:
    zi = _get_zoneinfo(tz_name)
    local = dt.astimezone(zi)
    return {
        "iso": local.isoformat(),
        "date": local.strftime("%A, %B %-d, %Y"),
        "time": local.strftime("%I:%M %p"),
        "time24": local.strftime("%H:%M"),
        "year": local.year,
        "month": local.strftime("%B"),
        "monthNumber": local.month,
        "day": local.day,
        "dayOfWeek": local.strftime("%A"),
        "timezone": tz_name,
        "timezoneAbbr": local.strftime("%Z"),
    }


def _current_time(tz_name: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "action": "current",
        **_format_datetime(now, tz_name),
        "isServerLocalTime": tz_name == SERVER_TIMEZONE,
        "serverTimezone": SERVER_TIMEZONE,
    }


def _convert_time(from_tz: str, to_tz: str, date_str: str | None) -> dict:
    if date_str:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_get_zoneinfo(from_tz))
    else:
        dt = datetime.now(timezone.utc)
    return {
        "action": "convert",
        "from": _format_datetime(dt, from_tz),
        "to": _format_datetime(dt, to_tz),
    }


def _calculate_date(
    base_date: str | None,
    amount: float,
    unit: str,
    operation: str,
    tz_name: str,
) -> dict:
    dt = (
        datetime.fromisoformat(base_date)
        if base_date
        else datetime.now(timezone.utc)
    )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_get_zoneinfo(tz_name))

    sign = -1 if operation == "subtract" else 1
    n = int(amount) * sign

    match unit:
        case "minutes":
            dt += timedelta(minutes=n)
        case "hours":
            dt += timedelta(hours=n)
        case "days":
            dt += timedelta(days=n)
        case "weeks":
            dt += timedelta(weeks=n)
        case "months":
            # Approximate: add calendar months
            month = dt.month - 1 + n
            year = dt.year + month // 12
            month = month % 12 + 1
            day = min(dt.day, [31,29,31,30,31,30,31,31,30,31,30,31][month - 1])
            dt = dt.replace(year=year, month=month, day=day)
        case "years":
            dt = dt.replace(year=dt.year + n)
        case _:
            raise ValueError(f"Unknown unit: {unit!r}")

    return {
        "action": "calculate",
        "operation": operation,
        "amount": amount,
        "unit": unit,
        "result": _format_datetime(dt, tz_name),
    }


def _date_difference(date1_str: str | None, date2_str: str, tz_name: str) -> dict:
    dt1 = (
        datetime.fromisoformat(date1_str)
        if date1_str
        else datetime.now(timezone.utc)
    )
    dt2 = datetime.fromisoformat(date2_str)

    zi = _get_zoneinfo(tz_name)
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=zi)
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=zi)

    diff = dt2 - dt1
    total_seconds = int(diff.total_seconds())
    total_days = diff.days
    total_hours = total_seconds // 3600
    total_minutes = total_seconds // 60

    abs_days = abs(total_days)
    weeks = abs_days // 7
    remaining_days = abs_days % 7
    is_past = total_seconds < 0

    return {
        "action": "difference",
        "from": _format_datetime(dt1, tz_name),
        "to": _format_datetime(dt2, tz_name),
        "difference": {
            "totalDays": total_days,
            "totalHours": total_hours,
            "totalMinutes": total_minutes,
            "weeks": weeks,
            "remainingDays": remaining_days,
            "isPast": is_past,
            "humanReadable": (
                f"{abs_days} days ago" if is_past else f"{abs_days} days from now"
            ),
        },
    }


class DateTimeTool(Tool):
    name = "getDateAndTimeTool"
    description = (
        "Get current date/time, convert between timezones, calculate future/past dates, "
        "or find the difference between dates.\n\n"
        "Actions:\n"
        '- "current": Get current time in a timezone (default: server local timezone).\n'
        '- "convert": Convert a time between two timezones.\n'
        '- "calculate": Add or subtract time from a date.\n'
        '- "difference": Find days/time between two dates.\n\n'
        "Supported timezones: PST, EST, CST, MST, UTC, GMT, Tokyo, Seoul, Beijing, "
        "Singapore, Sydney, London, Paris, Berlin, Mumbai, Dubai, Moscow, or any IANA timezone."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["current", "convert", "calculate", "difference"],
                "description": "The operation to perform.",
            },
            "timezone": {
                "type": "string",
                "description": (
                    'Timezone for current time or calculations (e.g. "Tokyo", "PST", '
                    '"America/New_York"). Defaults to server local timezone.'
                ),
            },
            "fromTimezone": {
                "type": "string",
                "description": "Source timezone for conversion.",
            },
            "toTimezone": {
                "type": "string",
                "description": "Target timezone for conversion.",
            },
            "date": {
                "type": "string",
                "description": "Base date (ISO format, e.g. '2024-12-25'). Defaults to now.",
            },
            "targetDate": {
                "type": "string",
                "description": "Target date for difference calculation (e.g. '2024-12-25').",
            },
            "amount": {
                "type": "number",
                "description": "Amount of time units to add/subtract.",
            },
            "unit": {
                "type": "string",
                "enum": ["minutes", "hours", "days", "weeks", "months", "years"],
                "description": "Time unit for calculations.",
            },
            "operation": {
                "type": "string",
                "enum": ["add", "subtract"],
                "description": "Whether to add or subtract time.",
            },
        },
        "required": [],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        action = params.get("action") or "current"
        try:
            match action:
                case "current":
                    tz = _resolve_timezone(params.get("timezone"))
                    return _current_time(tz)

                case "convert":
                    from_tz = _resolve_timezone(
                        params.get("fromTimezone") or params.get("timezone")
                    )
                    to_tz = _resolve_timezone(params.get("toTimezone"))
                    return _convert_time(from_tz, to_tz, params.get("date"))

                case "calculate":
                    amount = params.get("amount")
                    unit = params.get("unit")
                    if amount is None or not unit:
                        raise ValueError("calculate action requires 'amount' and 'unit'")
                    tz = _resolve_timezone(params.get("timezone"))
                    return _calculate_date(
                        params.get("date"),
                        float(amount),
                        unit,
                        params.get("operation") or "add",
                        tz,
                    )

                case "difference":
                    if not params.get("targetDate"):
                        raise ValueError("difference action requires 'targetDate'")
                    tz = _resolve_timezone(params.get("timezone"))
                    return _date_difference(params.get("date"), params["targetDate"], tz)

                case _:
                    raise ValueError(f"Unknown action: {action!r}")

        except Exception as exc:
            return {"error": True, "message": str(exc), "action": action}


date_time_tool = DateTimeTool()
