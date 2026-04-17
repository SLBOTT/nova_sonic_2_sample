"""
LocationSearchTool — searches for location coordinates via Open-Meteo Geocoding API.
Use this to resolve a city/country name to lat/lon before calling the WeatherTool.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .base import Tool

_HEADERS = {"User-Agent": "NovaSonicVoicebot/1.0", "Accept": "application/json"}
_TIMEOUT = 10.0


async def _search(query: str, count: int) -> dict:
    url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={quote(query)}&count={count}&language=en"
    )
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    results = [
        {
            "name": item["name"],
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "country": item.get("country", ""),
        }
        for item in (data.get("results") or [])
    ]
    return {"locations": results}


class LocationSearchTool(Tool):
    name = "searchLocationTool"
    description = (
        "Search for a city or country to get its coordinates (latitude/longitude). "
        "Query must be in English. Use this to find coordinates before getting weather data."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "City name, country name, or postal code to search for. Must be in English.",
            },
            "count": {
                "type": "number",
                "description": "Number of results to return (1-10). Default is 3.",
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        query = (params.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        count = max(1, min(int(params.get("count") or 3), 10))

        print(f'LocationSearchTool: searching "{query}" (count={count})', flush=True)
        return await _search(query, count)


location_search_tool = LocationSearchTool()
