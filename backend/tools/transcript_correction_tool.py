"""
TranscriptCorrectionTool — auto-corrects speech recognition errors using a Bedrock
Converse model. Suggests phonetically similar real-world terms for misheard words.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import boto3

from ..consts import TOOL_MODELS
from .base import Tool

_SYSTEM_PROMPT = (
    "You are a speech-to-text correction assistant specialised in identifying "
    "misheard terms from audio transcripts."
)

# boto3 clients are thread-safe; cache one per region
_clients: dict[str, Any] = {}


def _get_client(region: str) -> Any:
    if region not in _clients:
        _clients[region] = boto3.client("bedrock-runtime", region_name=region)
    return _clients[region]


def _format_conversation_context(conversations: list[dict] | None) -> str:
    if not conversations:
        return "No recent conversation context available."
    return "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversations[-3:]
    )


def _invoke_sync(
    unclear_term: str,
    conversations: list[dict] | None,
    inference_config: dict | None,
) -> dict:
    cfg = TOOL_MODELS["transcript_correction"]
    client = _get_client(cfg["region"])

    conversation_context = _format_conversation_context(conversations)

    user_message = (
        f'In an audio transcript, users mention "{unclear_term}" and I believe there '
        "are some transcript problems based on similar phonetics or pronunciation. "
        "Consider the pronunciations for words with similar phonetics. "
        "Suggest a list of possible corrections for this term.\n\n"
        "## CONSIDERATIONS ##\n"
        "- The results MUST have similar pronunciations. Remove any results that do not share similar pronunciations.\n"
        "- Consider the conversation for more background.\n"
        "- Look for popular and well-known terms.\n"
        "- Results must be meaningful and human-understandable.\n\n"
        "## Output ##\n"
        "List a bullet point of at most 5 most-likely possible words without any other texts or explanations\n\n"
        f"<conversation>{conversation_context}</conversation>"
    )

    temperature = (inference_config or {}).get("temperature", 0.3)
    top_p = (inference_config or {}).get("topP", 0.9)

    command_input: dict[str, Any] = {
        "modelId": cfg["model_id"],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "system": [{"text": _SYSTEM_PROMPT}],
        "inferenceConfig": {
            "maxTokens": 1024,
            "temperature": temperature,
            "topP": top_p,
        },
    }

    is_nova = "nova" in cfg["model_id"]

    if is_nova and cfg.get("web_grounding"):
        command_input["toolConfig"] = {
            "tools": [{"systemTool": {"name": "nova_grounding"}}]
        }

    if is_nova and cfg.get("extended_thinking"):
        command_input["additionalModelRequestFields"] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": cfg.get("max_reasoning_effort", "low"),
            }
        }

    response = client.converse(**command_input)
    output_text = (
        response.get("output", {})
        .get("message", {})
        .get("content", [{}])[0]
        .get("text", "")
    )

    print(f"TranscriptCorrectionTool: suggestions for {unclear_term!r}", flush=True)
    return {"unclearTerm": unclear_term, "suggestions": output_text}


class TranscriptCorrectionTool(Tool):
    name = "transcriptCorrectionTool"
    description = (
        "Fixes speech recognition errors by analysing phonetic similarities. "
        "Use this when a user corrects you, repeats themselves, or when a name, place, "
        "or term does not match any known entity. Also use it when proper nouns seem "
        "misspelled or the conversation context suggests a different word than what was "
        "transcribed. Input the unclear term and recent conversation for context. "
        "The tool returns likely corrections based on similar pronunciations."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "unclearTerm": {
                "type": "string",
                "description": "The unclear or potentially misheard term that needs correction.",
            },
            "conversations": {
                "type": "array",
                "description": "Recent conversation messages (last 3) for context.",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant"],
                            "description": "Who said this message.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The message content.",
                        },
                    },
                    "required": ["role", "content"],
                },
            },
        },
        "required": ["unclearTerm"],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        unclear_term = (params.get("unclearTerm") or "").strip()
        if not unclear_term:
            return {"error": True, "message": "unclearTerm is required"}

        conversations = params.get("conversations")

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                partial(_invoke_sync, unclear_term, conversations, inference_config),
            )
        except Exception as exc:
            return {"error": True, "message": str(exc), "unclearTerm": unclear_term}


transcript_correction_tool = TranscriptCorrectionTool()
