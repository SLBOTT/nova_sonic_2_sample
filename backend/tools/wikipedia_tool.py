"""
WikipediaTool — search and retrieve content from Wikipedia.
Uses Wikipedia's action=query API which handles multilingual queries.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, quote as url_quote

import httpx

from .base import Tool

MAX_CONTENT_LENGTH = 9_000  # chars; leave headroom for tool result limit

_HEADERS = {"User-Agent": "NovaSonicVoicebot/1.0", "Accept": "application/json"}
_TIMEOUT = 15.0


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


async def _search(query: str, limit: int) -> list[dict]:
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&prop=info&origin=*"
        f"&srlimit={limit}&utf8=&format=json"
        f"&srsearch={quote(query)}"
    )
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    return [
        {
            "title": item["title"],
            "snippet": _strip_html(item["snippet"]),
            "url": f"https://en.wikipedia.org/wiki/{quote(item['title'])}",
        }
        for item in (data.get("query", {}).get("search") or [])
    ]


async def _summary(query: str) -> dict:
    results = await _search(query, 1)
    if not results:
        raise ValueError(f'No Wikipedia articles found for "{query}"')

    title = results[0]["title"]
    encoded = quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)

    if not r.is_success:
        return {
            "title": title,
            "extract": results[0]["snippet"],
            "url": results[0]["url"],
        }

    data = r.json()
    return {
        "title": data.get("title", title),
        "extract": data.get("extract") or "No summary available.",
        "url": (
            data.get("content_urls", {}).get("desktop", {}).get("page")
            or results[0]["url"]
        ),
    }


async def _content(query: str) -> dict:
    results = await _search(query, 1)
    if not results:
        raise ValueError(f'No Wikipedia articles found for "{query}"')

    title = results[0]["title"]
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={quote(title)}"
        "&prop=extracts&explaintext=1&exsectionformat=plain&format=json&origin=*"
    )

    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract") or results[0]["snippet"]

    if len(extract) > MAX_CONTENT_LENGTH:
        extract = extract[:MAX_CONTENT_LENGTH] + "..."

    return {
        "title": page.get("title", title),
        "content": extract,
        "url": f"https://en.wikipedia.org/wiki/{url_quote(title)}",
    }


class WikipediaTool(Tool):
    name = "searchWikipedia"
    description = (
        "Look up factual information on Wikipedia. Use this for questions about people, "
        "places, events, science, technology, companies, movies, music, books, or any "
        "topic requiring accurate details. Your training data may be outdated, so verify "
        "facts here first. Use search mode to find relevant articles, summary mode for "
        "quick facts, or content mode for detailed information. For non-English names or "
        "terms, preserve the original characters without encoding."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. MUST preserve original script for non-English terms "
                    "(Chinese, Japanese, Korean, etc.). Do NOT romanize."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["search", "summary", "content"],
                "description": (
                    '"search": list matching articles; '
                    '"summary": brief 2-3 sentence overview; '
                    '"content": full detailed article text.'
                ),
            },
            "limit": {
                "type": "number",
                "description": "Max search results (1-10, default 5). Only for search mode.",
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        query = (params.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        mode = params.get("mode") or "search"
        limit = max(1, min(int(params.get("limit") or 5), 10))

        print(
            f"WikipediaTool: {mode} — {query!r}"
            + (f" (limit={limit})" if mode == "search" else ""),
            flush=True,
        )

        match mode:
            case "summary":
                return await _summary(query)
            case "content":
                return await _content(query)
            case _:
                return {"results": await _search(query, limit)}


wikipedia_tool = WikipediaTool()
