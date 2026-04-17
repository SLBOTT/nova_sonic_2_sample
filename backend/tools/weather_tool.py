"""
WeatherTool — fetches weather data from the Open-Meteo API (no API key required).
"""
from __future__ import annotations

from typing import Any

import httpx

from .base import Tool

# WMO Weather interpretation codes
WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_HEADERS = {"User-Agent": "NovaSonicVoicebot/1.0", "Accept": "application/json"}
_TIMEOUT = 10.0


async def _fetch_current(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,cloud_cover,weather_code"
        f"&timezone=auto"
    )
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    c = data["current"]
    return {
        "location": {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "timezone": data["timezone"],
        },
        "current": {
            "temperature": c["temperature_2m"],
            "humidity": c["relative_humidity_2m"],
            "cloud_cover": c["cloud_cover"],
            "conditions": WEATHER_CODES.get(c["weather_code"], "Unknown"),
        },
        "units": {"temperature": "°C", "humidity": "%", "cloud_cover": "%"},
    }


async def _fetch_forecast(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        f"relative_humidity_2m_max,relative_humidity_2m_min"
        f"&forecast_days=7&timezone=auto"
    )
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    daily = data["daily"]
    return {
        "location": {
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "timezone": data["timezone"],
        },
        "forecast": [
            {
                "date": daily["time"][i],
                "conditions": WEATHER_CODES.get(daily["weather_code"][i], "Unknown"),
                "temp_high": daily["temperature_2m_max"][i],
                "temp_low": daily["temperature_2m_min"][i],
                "humidity_high": daily["relative_humidity_2m_max"][i],
                "humidity_low": daily["relative_humidity_2m_min"][i],
            }
            for i in range(len(daily["time"]))
        ],
        "units": {"temperature": "°C", "humidity": "%"},
    }


class WeatherTool(Tool):
    name = "getWeatherTool"
    description = (
        'Get weather for a location. Use mode "current" for current conditions '
        'or "forecast" for a 7-day forecast.'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "string",
                "description": "Geographical WGS84 latitude of the location.",
            },
            "longitude": {
                "type": "string",
                "description": "Geographical WGS84 longitude of the location.",
            },
            "mode": {
                "type": "string",
                "enum": ["current", "forecast"],
                "description": (
                    '"current" for current conditions (default), '
                    '"forecast" for 7-day forecast.'
                ),
            },
        },
        "required": ["latitude", "longitude"],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        lat_raw = params.get("latitude")
        lon_raw = params.get("longitude")

        if lat_raw is None or lon_raw is None:
            raise ValueError("latitude and longitude are required")

        lat = float(lat_raw)
        lon = float(lon_raw)
        mode = params.get("mode") or "current"

        print(f"WeatherTool: fetching {mode} weather for ({lat}, {lon})", flush=True)

        if mode == "forecast":
            return await _fetch_forecast(lat, lon)
        return await _fetch_current(lat, lon)


weather_tool = WeatherTool()
