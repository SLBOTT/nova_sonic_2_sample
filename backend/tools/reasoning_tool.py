"""
ReasoningTool — delegates complex reasoning to a more powerful Bedrock model
via the Converse API (boto3).
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import boto3

from ..consts import TOOL_MODELS
from .base import Tool

_TASK_PROMPTS: dict[str, str] = {
    "reason": "Task: Break down the problem step by step using logical reasoning. Show your work clearly.",
    "analyze": "Task: Provide thorough analysis considering multiple perspectives, trade-offs, and implications.",
    "solve": "Task: Focus on finding practical solutions with actionable steps and recommendations.",
    "explain": "Task: Explain concepts clearly with examples and analogies, suitable for learning.",
    "verify": "Task: Fact-check and verify the information. Point out any errors, misconceptions, or areas of uncertainty.",
    "brainstorm": "Task: Generate creative ideas and alternative approaches. Think outside the box.",
    "summarize": "Task: Distill the key points into a clear, organised summary.",
}

_BASE_SYSTEM_PROMPT = (
    "You are an advanced reasoning assistant embedded within a voice-based AI system. "
    "Your role is to provide deeper analysis, fact-checking, and complex problem-solving "
    "support when the primary voice assistant needs backup.\n\n"
    "Be accurate and thoughtful since the voice assistant is relying on you for correctness. "
    "Acknowledge uncertainty when appropriate. Provide structured thinking when helpful, "
    "but keep it concise.\n\n"
    "Never make up facts, statistics, dates, names, or any specific data. If you do not "
    "know something with certainty, say you are not sure or that you do not have that "
    "information. When uncertain, clearly state your confidence level. Prefer saying you "
    "do not know over providing potentially false information.\n\n"
)

_VOICE_REMINDER = (
    "\n\nIMPORTANT: Keep your response concise (2-4 sentences) since this will be spoken "
    "aloud. Be direct and conversational."
)

# boto3 clients are thread-safe; cache one per region
_clients: dict[str, Any] = {}


def _get_client(region: str) -> Any:
    if region not in _clients:
        _clients[region] = boto3.client("bedrock-runtime", region_name=region)
    return _clients[region]


def _invoke_sync(
    question: str,
    context: str | None,
    task: str | None,
    inference_config: dict | None,
) -> dict:
    cfg = TOOL_MODELS["reasoning"]
    client = _get_client(cfg["region"])

    task_prompt = _TASK_PROMPTS.get(task or "", "Task: Provide a comprehensive, well-reasoned response with your best thinking.")
    system_prompt = _BASE_SYSTEM_PROMPT + task_prompt + _VOICE_REMINDER

    user_message = question
    if context:
        user_message = f"Context: {context}\n\nQuestion: {question}"

    temperature = (inference_config or {}).get("temperature", 0.7)
    top_p = (inference_config or {}).get("topP", 0.9)

    command_input: dict[str, Any] = {
        "modelId": cfg["model_id"],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "maxTokens": 2048,
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

    content_list = response.get("output", {}).get("message", {}).get("content", [])
    output_text = ""
    reasoning_text = ""

    for item in content_list:
        if "reasoningContent" in item:
            reasoning_text = item["reasoningContent"].get("reasoningText", {}).get("text", "")
        elif "text" in item:
            output_text += item["text"]

    if not output_text:
        output_text = "No response generated"

    print(f"ReasoningTool: response ({len(output_text)} chars)", flush=True)

    result: dict = {"answer": output_text}
    if reasoning_text:
        result["reasoning"] = reasoning_text
    return result


class ReasoningTool(Tool):
    name = "reasoningTool"
    description = (
        "Use this tool for complex reasoning, fact-checking, and deep thinking. "
        "It calls a more powerful reasoning model for challenging questions, complex "
        "math or logic problems, multi-step analysis, pros/cons comparisons, creative "
        "brainstorming, or when you want to verify your answer. "
        "Better to be accurate than fast."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question, problem, or topic to think deeply about.",
            },
            "context": {
                "type": "string",
                "description": "Relevant background info, conversation history, or constraints.",
            },
            "task": {
                "type": "string",
                "enum": ["reason", "analyze", "solve", "explain", "verify", "brainstorm", "summarize"],
                "description": (
                    "Type of thinking needed: reason (step-by-step logic), "
                    "analyze (multi-perspective), solve (find solutions), "
                    "explain (teach clearly), verify (fact-check), "
                    "brainstorm (creative ideas), summarize (distill key points)."
                ),
            },
        },
        "required": ["question"],
    }

    async def execute(self, params: dict, inference_config: dict | None = None) -> Any:
        question = (params.get("question") or "").strip()
        if not question:
            return {"error": True, "message": "A question is required for the reasoning model"}

        context = params.get("context")
        task = params.get("task")

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                partial(_invoke_sync, question, context, task, inference_config),
            )
        except Exception as exc:
            return {"error": True, "message": str(exc), "question": question}


reasoning_tool = ReasoningTool()
