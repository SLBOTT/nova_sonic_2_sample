import argparse
import asyncio
import json
import os
from pathlib import Path
import uuid

from dotenv import load_dotenv

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity.environment import (
    EnvironmentCredentialsResolver,
)


DEFAULT_MODEL_ID = "amazon.nova-2-sonic-v1:0"
DEFAULT_REGION = "us-east-1"


def make_client(region: str) -> BedrockRuntimeClient:
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
    )
    return BedrockRuntimeClient(config=config)


async def send_event(stream, event: dict) -> None:
    event_json = json.dumps(event)
    print(f">>> {event_json}", flush=True)
    chunk = InvokeModelWithBidirectionalStreamInputChunk(
        value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
    )
    await stream.input_stream.send(chunk)
    await asyncio.sleep(0.05)


async def receive_events(stream, timeout_seconds: int) -> bool:
    saw_model_output = False
    saw_stream_event = False
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while asyncio.get_running_loop().time() < deadline:
        try:
            output = await asyncio.wait_for(stream.await_output(), timeout=2)
            if output is None:
                continue
            result = await asyncio.wait_for(output[1].receive(), timeout=2)
        except asyncio.TimeoutError:
            continue

        if result is None:
            continue

        if not result.value or not result.value.bytes_:
            continue

        payload = result.value.bytes_.decode("utf-8")
        print(f"<<< {payload}", flush=True)
        saw_stream_event = True

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        event = data.get("event", {})
        if event.get("textOutput") or event.get("audioOutput") or event.get("completionStart"):
            saw_model_output = True

        if event.get("completionEnd"):
            return saw_model_output

    if saw_stream_event:
        print("Received Bedrock stream events, but no text/audio output before timeout.", flush=True)

    return saw_model_output


async def run_probe(args: argparse.Namespace) -> int:
    client = make_client(args.region)
    prompt_name = str(uuid.uuid4())
    system_content = str(uuid.uuid4())
    user_content = str(uuid.uuid4())

    print(f"Opening stream: model={args.model_id}, region={args.region}", flush=True)
    stream = await asyncio.wait_for(
        client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=args.model_id)
        ),
        timeout=args.connect_timeout,
    )

    receive_task = asyncio.create_task(receive_events(stream, args.timeout))

    try:
        await send_event(
            stream,
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": 256,
                            "topP": 0.9,
                            "temperature": 0.7,
                        }
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "promptStart": {
                        "promptName": prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 24000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": "tiffany",
                        },
                        "toolUseOutputConfiguration": {"mediaType": "application/json"},
                        "toolConfiguration": {"tools": [], "toolChoice": {"auto": {}}},
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": system_content,
                        "type": "TEXT",
                        "interactive": False,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "textInput": {
                        "promptName": prompt_name,
                        "contentName": system_content,
                        "content": "You are a concise assistant. Reply briefly.",
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "contentEnd": {
                        "promptName": prompt_name,
                        "contentName": system_content,
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": user_content,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "textInput": {
                        "promptName": prompt_name,
                        "contentName": user_content,
                        "content": args.message,
                    }
                }
            },
        )
        await send_event(
            stream,
            {
                "event": {
                    "contentEnd": {
                        "promptName": prompt_name,
                        "contentName": user_content,
                    }
                }
            },
        )

        saw_model_output = await receive_task
        return 0 if saw_model_output else 2
    finally:
        for event in (
            {"event": {"promptEnd": {"promptName": prompt_name}}},
            {"event": {"sessionEnd": {}}},
        ):
            try:
                await send_event(stream, event)
            except Exception as exc:
                print(f"Cleanup send failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Python support for Bedrock Nova Sonic bidirectional streaming."
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", DEFAULT_REGION),
        help="AWS region for Bedrock Runtime.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("NOVA_SONIC_MODEL_ID", DEFAULT_MODEL_ID),
        help="Nova Sonic model ID to probe.",
    )
    parser.add_argument(
        "--message",
        default="Say hello in one short sentence.",
        help="Text message to send through the bidirectional stream.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Seconds to wait for model output.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="Seconds to wait while opening the Bedrock bidirectional stream.",
    )
    return parser.parse_args()


def main() -> int:
    backend_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(backend_env)
    load_dotenv()
    args = parse_args()
    return asyncio.run(run_probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
