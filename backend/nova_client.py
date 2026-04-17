import asyncio
import base64
import json
import re
import uuid
from collections.abc import Awaitable, Callable

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
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from tools.base import ToolRegistry

EventCallback = Callable[[str, dict], Awaitable[None]]

MAX_TOOL_RESULT_LENGTH = 20_480

DEFAULT_MODEL_ID = "amazon.nova-2-sonic-v1:0"


def make_bedrock_client(region: str) -> BedrockRuntimeClient:
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
    )
    return BedrockRuntimeClient(config=config)


class NovaSonicSession:
    def __init__(
        self,
        *,
        session_id: str,
        region: str,
        model_id: str = DEFAULT_MODEL_ID,
        inference_config: dict | None = None,
        turn_detection_config: dict | None = None,
        enabled_tools: list[str] | None = None,
        tool_registry: ToolRegistry | None = None,
        on_event: EventCallback,
    ) -> None:
        self.session_id = session_id
        self.region = region
        self.model_id = model_id
        self.inference_config = inference_config or {
            "maxTokens": 1024,
            "topP": 0.9,
            "temperature": 0.7,
        }
        self.turn_detection_config = turn_detection_config or {}
        self.enabled_tools = enabled_tools  # None means all tools
        self.tool_registry = tool_registry or ToolRegistry()
        self.on_event = on_event

        self.client = make_bedrock_client(region)
        self.stream = None
        self.receiver_task: asyncio.Task | None = None
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.active = False
        self.prompt_started = False
        self.audio_started = False
        self.send_lock = asyncio.Lock()
        # Pending tool use fields (populated on toolUse event, consumed on contentEnd TOOL)
        self._pending_tool_use_id: str = ""
        self._pending_tool_name: str = ""
        self._pending_tool_content: dict | None = None

    async def open(self) -> None:
        if self.stream:
            return

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.active = True
        self.receiver_task = asyncio.create_task(self._receive_loop())

    async def setup_prompt_start(
        self,
        *,
        voice_id: str = "tiffany",
        output_sample_rate: int = 24000,
    ) -> None:
        await self.open()

        session_start = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": self.inference_config,
                }
            }
        }
        endpointing_sensitivity = self.turn_detection_config.get(
            "endpointingSensitivity"
        )
        if endpointing_sensitivity:
            session_start["event"]["sessionStart"]["turnDetectionConfiguration"] = {
                "endpointingSensitivity": endpointing_sensitivity
            }

        await self.send_event(session_start)
        # Build tool specs filtered by enabled_tools if specified
        all_specs = self.tool_registry.get_tool_specs()
        if self.enabled_tools is not None:
            enabled_lower = {t.lower() for t in self.enabled_tools}
            all_specs = [
                s for s in all_specs
                if s["toolSpec"]["name"].lower() in enabled_lower
            ]
        print(
            f"Session {self.session_id[:8]}: {len(all_specs)} tool(s) configured: "
            + ", ".join(s["toolSpec"]["name"] for s in all_specs),
            flush=True,
        )

        await self.send_event(
            {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": output_sample_rate,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": voice_id,
                        },
                        "toolUseOutputConfiguration": {"mediaType": "application/json"},
                        "toolConfiguration": {
                            "tools": all_specs,
                            "toolChoice": {"auto": {}},
                        },
                    }
                }
            }
        )
        self.prompt_started = True

    async def send_system_prompt(self, content: str) -> None:
        prompt = (content or "You are a helpful conversational assistant.").strip()
        content_name = str(uuid.uuid4())
        await self._send_text_content(
            content_name=content_name,
            role="SYSTEM",
            interactive=False,
            content=prompt,
        )

    async def start_audio(self) -> None:
        await self.send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "audioType": "SPEECH",
                            "encoding": "base64",
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                        },
                    }
                }
            }
        )
        self.audio_started = True

    async def send_audio_input(self, audio_base64: str | bytes) -> None:
        if isinstance(audio_base64, bytes):
            audio_base64 = base64.b64encode(audio_base64).decode("utf-8")

        await self.send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": audio_base64,
                    }
                }
            }
        )

    async def send_text_input(self, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        await self._send_text_content(
            content_name=str(uuid.uuid4()),
            role="USER",
            interactive=True,
            content=text,
        )

    async def close(self) -> None:
        if not self.active:
            return

        if self.audio_started:
            try:
                await self.send_event(
                    {
                        "event": {
                            "contentEnd": {
                                "promptName": self.prompt_name,
                                "contentName": self.audio_content_name,
                            }
                        }
                    }
                )
            except Exception as exc:
                print(f"Error sending audio contentEnd: {exc}", flush=True)
            await asyncio.sleep(0.2)

        if self.prompt_started:
            try:
                await self.send_event(
                    {"event": {"promptEnd": {"promptName": self.prompt_name}}}
                )
            except Exception as exc:
                print(f"Error sending promptEnd: {exc}", flush=True)
            await asyncio.sleep(0.2)

        try:
            await self.send_event({"event": {"sessionEnd": {}}})
        except Exception as exc:
            print(f"Error sending sessionEnd: {exc}", flush=True)
        await asyncio.sleep(0.2)
        self.active = False

        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await asyncio.wait_for(self.receiver_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            finally:
                self.receiver_task = None

    async def _send_text_content(
        self,
        *,
        content_name: str,
        role: str,
        interactive: bool,
        content: str,
    ) -> None:
        await self.send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": interactive,
                        "role": role,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self.send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": content,
                    }
                }
            }
        )
        await self.send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )

    async def send_event(self, event: dict) -> None:
        if not self.stream:
            raise RuntimeError("Nova stream is not open")

        event_json = json.dumps(event)
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )

        async with self.send_lock:
            await self.stream.input_stream.send(chunk)

    async def _receive_loop(self) -> None:
        while self.active and self.stream:
            try:
                output = await self.stream.await_output()
                if output is None:
                    continue

                result = await output[1].receive()
                if result is None or not result.value or not result.value.bytes_:
                    continue

                payload = result.value.bytes_.decode("utf-8")
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    await self.on_event("unknown", {"raw": payload})
                    continue

                await self._dispatch_response(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                try:
                    await self.on_event(
                        "error",
                        {
                            "source": "responseStream",
                            "details": str(exc),
                        },
                    )
                except Exception:
                    pass
                self.active = False

    async def send_tool_result(self, tool_use_id: str, result: object) -> None:
        """Send a tool result back to Nova Sonic as a TOOL content block."""
        content_id = str(uuid.uuid4())

        result_str = result if isinstance(result, str) else json.dumps(result)
        # Strip control characters (keep \t \n \r)
        result_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", result_str)
        if len(result_str) > MAX_TOOL_RESULT_LENGTH:
            print(
                f"[ToolResult] Truncating from {len(result_str)} to {MAX_TOOL_RESULT_LENGTH} chars",
                flush=True,
            )
            result_str = result_str[:MAX_TOOL_RESULT_LENGTH] + "... (truncated)"

        print(f"[ToolResult] Sending {len(result_str)} chars for toolUseId={tool_use_id[:8]}...", flush=True)

        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_id,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        })

        await asyncio.sleep(0.05)

        await self.send_event({
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_id,
                    "content": result_str,
                }
            }
        })

        await asyncio.sleep(0.05)

        await self.send_event({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": content_id,
                }
            }
        })

        await asyncio.sleep(0.1)
        print(f"[ToolResult] ✓ Sent for toolUseId={tool_use_id[:8]}...", flush=True)

    async def _execute_tool_async(self, tool_use_id: str, tool_name: str, tool_content: dict) -> None:
        """Execute a tool asynchronously without blocking the receive loop."""
        start = asyncio.get_event_loop().time()
        print(f"\n┌─── Tool Invocation ───────────────────────────", flush=True)
        print(f"│ Tool: {tool_name}", flush=True)
        print(f"│ Session: {self.session_id[:8]}...", flush=True)
        print(f"│ Tool Use ID: {tool_use_id}", flush=True)
        print(f"└───────────────────────────────────────────────", flush=True)

        if not self.tool_registry.has(tool_name):
            print(f"[Tool:{tool_name}] ❌ Not found in registry", flush=True)
            error_result = {"error": True, "message": f'Tool "{tool_name}" not supported'}
            if self.active:
                await self.send_tool_result(tool_use_id, error_result)
            return

        try:
            # Nova Sonic sends tool params as a JSON string inside content.content
            params: dict = tool_content
            raw_content = tool_content.get("content")
            if isinstance(raw_content, str):
                try:
                    params = json.loads(raw_content)
                    print(f"[Tool:{tool_name}] Parsed params: {json.dumps(params)[:200]}", flush=True)
                except json.JSONDecodeError:
                    params = {"content": raw_content}

            result = await self.tool_registry.execute(
                tool_name, params, self.inference_config
            )
            duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)

            print(f"┌─── Tool Result ───────────────────────────────", flush=True)
            print(f"│ Tool: {tool_name}", flush=True)
            print(f"│ Status: ✓ Success", flush=True)
            print(f"│ Duration: {duration_ms}ms", flush=True)
            print(f"└───────────────────────────────────────────────\n", flush=True)

            if self.active:
                await self.send_tool_result(tool_use_id, result)
                await self.on_event("toolResult", {
                    "toolUseId": tool_use_id,
                    "result": result,
                    "executionTimeMs": duration_ms,
                })

        except Exception as exc:
            duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)
            error_msg = str(exc)
            print(f"┌─── Tool Result ───────────────────────────────", flush=True)
            print(f"│ Tool: {tool_name}", flush=True)
            print(f"│ Status: ❌ Failed", flush=True)
            print(f"│ Error: {error_msg}", flush=True)
            print(f"│ Duration: {duration_ms}ms", flush=True)
            print(f"└───────────────────────────────────────────────\n", flush=True)

            error_result = {"error": True, "message": error_msg}
            if self.active:
                await self.send_tool_result(tool_use_id, error_result)
                await self.on_event("toolResult", {
                    "toolUseId": tool_use_id,
                    "result": error_result,
                    "executionTimeMs": duration_ms,
                    "error": True,
                })

    async def _dispatch_response(self, data: dict) -> None:
        event = data.get("event")
        if not event:
            await self.on_event("unknown", data)
            return

        if "toolUse" in event:
            # Store pending tool use; execution is triggered on contentEnd with type=TOOL
            self._pending_tool_use_id = event["toolUse"].get("toolUseId", "")
            self._pending_tool_name = event["toolUse"].get("toolName", "")
            self._pending_tool_content = event["toolUse"]
            await self.on_event("toolUse", event["toolUse"])
            return

        if "contentEnd" in event and event["contentEnd"].get("type") == "TOOL":
            # Fire-and-forget: execute the tool without blocking the receive loop
            tool_use_id = self._pending_tool_use_id
            tool_name = self._pending_tool_name
            tool_content = self._pending_tool_content or {}
            asyncio.create_task(
                self._execute_tool_async(tool_use_id, tool_name, tool_content)
            )
            await self.on_event("toolEnd", {
                "toolUseId": tool_use_id,
                "toolName": tool_name,
                "toolUseContent": tool_content,
            })
            return

        event_name = next(iter(event.keys()), "unknown")
        await self.on_event(event_name, event[event_name])
