"""
BedrockKnowledgeBaseTool — retrieve relevant chunks from Amazon Bedrock Knowledge Bases.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import boto3

from .base import Tool

DEFAULT_TOP_K = 4
MIN_TOP_K = 1
MAX_TOP_K = 10


def _resolve_source(location: dict[str, Any] | None) -> str | None:
    if not location:
        return None

    if location.get("s3Location"):
        uri = location["s3Location"].get("uri")
        if uri:
            return uri

    if location.get("webLocation"):
        url = location["webLocation"].get("url")
        if url:
            return url

    if location.get("confluenceLocation"):
        url = location["confluenceLocation"].get("url")
        if url:
            return url

    if location.get("salesforceLocation"):
        url = location["salesforceLocation"].get("url")
        if url:
            return url

    if location.get("sharePointLocation"):
        url = location["sharePointLocation"].get("url")
        if url:
            return url

    if location.get("customDocumentLocation"):
        doc_id = location["customDocumentLocation"].get("id")
        if doc_id:
            return f"customDocument:{doc_id}"

    if location.get("kendraDocumentLocation"):
        uri = location["kendraDocumentLocation"].get("uri")
        if uri:
            return uri

    if location.get("sqlLocation"):
        query = location["sqlLocation"].get("query")
        if query:
            return f"sql:{query}"

    if location.get("type"):
        return str(location["type"])

    return None


class BedrockKnowledgeBaseTool(Tool):
    name = "searchKnowledgeBase"
    description = (
        "Search the configured Amazon Bedrock Knowledge Base and return relevant source "
        "chunks for grounding answers. Use this for organization-specific or private "
        "knowledge before answering."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question to search in the knowledge base.",
            },
            "maxResults": {
                "type": "number",
                "description": (
                    "Maximum number of chunks to return (1-10). "
                    "If omitted, BEDROCK_KB_TOP_K env var is used."
                ),
            },
        },
        "required": ["query"],
    }

    def _build_client(self, region: str):
        return boto3.client("bedrock-agent-runtime", region_name=region)

    def _retrieve_sync(
        self,
        region: str,
        kb_id: str,
        query: str,
        max_results: int,
        search_type: str | None,
    ) -> dict[str, Any]:
        client = self._build_client(region)
        vector_config: dict[str, Any] = {"numberOfResults": max_results}
        if search_type in {"HYBRID", "SEMANTIC"}:
            vector_config["overrideSearchType"] = search_type

        return client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector_config},
        )

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        query = (params.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        kb_id = os.getenv("BEDROCK_KB_ID", "").strip()
        if not kb_id:
            raise ValueError("BEDROCK_KB_ID is not set")

        region = os.getenv("AWS_REGION", "us-east-1")
        search_type = (os.getenv("BEDROCK_KB_SEARCH_TYPE") or "").strip().upper() or None

        default_top_k = int(os.getenv("BEDROCK_KB_TOP_K") or DEFAULT_TOP_K)
        raw_top_k = params.get("maxResults", default_top_k)
        max_results = max(MIN_TOP_K, min(int(raw_top_k), MAX_TOP_K))

        print(
            f"BedrockKnowledgeBaseTool: retrieve query={query!r}, topK={max_results}",
            flush=True,
        )

        response = await asyncio.to_thread(
            self._retrieve_sync,
            region,
            kb_id,
            query,
            max_results,
            search_type,
        )

        results = []
        for item in response.get("retrievalResults") or []:
            content = (item.get("content") or {}).get("text")
            if not content:
                continue

            source = _resolve_source(item.get("location"))
            results.append(
                {
                    "content": content,
                    "score": item.get("score"),
                    "source": source,
                    "metadata": item.get("metadata") or {},
                }
            )

        return {
            "query": query,
            "knowledgeBaseId": kb_id,
            "resultCount": len(results),
            "results": results,
        }


knowledge_base_tool = BedrockKnowledgeBaseTool()
