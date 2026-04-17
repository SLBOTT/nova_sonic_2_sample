import os

NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"

TOOL_MODELS = {
    "reasoning": {
        "model_id": os.getenv(
            "REASONING_MODEL_ID",
            "global.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        "region": os.getenv("REASONING_MODEL_REGION", "ap-northeast-1"),
        "extended_thinking": False,
        "max_reasoning_effort": "low",
        "web_grounding": False,
    },
    "transcript_correction": {
        "model_id": os.getenv(
            "TRANSCRIPT_CORRECTION_MODEL_ID",
            "global.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        "region": os.getenv("TRANSCRIPT_CORRECTION_MODEL_REGION", "ap-northeast-1"),
        "extended_thinking": False,
        "max_reasoning_effort": "low",
        "web_grounding": False,
    },
}

DEFAULT_INFERENCE_CONFIG = {
    "maxTokens": 1024,
    "topP": 0.9,
    "temperature": 0.7,
}

# Maximum length (chars) returned by any tool to keep LLM token usage reasonable
MAX_TOOL_RESULT_LENGTH = 10_000
