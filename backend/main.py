import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nova_client import DEFAULT_MODEL_ID, NovaSonicSession


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
PUBLIC_DIR = ROOT_DIR / "public"
DEFAULT_REGION = "us-east-1"

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(ROOT_DIR / ".env")

sessions: dict[str, NovaSonicSession] = {}
session_configs: dict[str, dict] = {}
connections: dict[str, WebSocket] = {}
connection_locks: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shut down all active sessions when the server exits
    sids = list(sessions.keys())
    if sids:
        print(f"Closing {len(sids)} active session(s) on shutdown...", flush=True)
        await asyncio.gather(
            *(close_session(sid) for sid in sids),
            return_exceptions=True,
        )


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/tools")
async def tools():
    return {"tools": []}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "activeSessions": len(sessions),
        "modelId": os.getenv("NOVA_SONIC_MODEL_ID", DEFAULT_MODEL_ID),
    }


async def send_ws(sid: str, message: dict[str, Any]) -> None:
    websocket = connections.get(sid)
    if not websocket:
        return

    lock = connection_locks.setdefault(sid, asyncio.Lock())
    async with lock:
        try:
            await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError, OSError):
            connections.pop(sid, None)
            connection_locks.pop(sid, None)


async def emit_to_socket(sid: str, event_name: str, data: dict | None = None) -> None:
    await send_ws(sid, {"event": event_name, "data": data})


async def send_ack(
    sid: str,
    message_id: str | None,
    data: dict | None = None,
) -> None:
    if message_id:
        await send_ws(sid, {"replyTo": message_id, "data": data or {}})


async def close_session(sid: str) -> None:
    session = sessions.pop(sid, None)
    session_configs.pop(sid, None)
    if session:
        try:
            await asyncio.wait_for(session.close(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"Session {sid} close timed out, forcing cleanup", flush=True)
            if session.receiver_task:
                session.receiver_task.cancel()
        except Exception as exc:
            print(f"Session cleanup error for {sid}: {exc}", flush=True)


async def handle_initialize_connection(
    sid: str,
    data: dict | None,
    message_id: str | None,
) -> None:
    config = data or {}
    if sid in sessions:
        await send_ack(sid, message_id, {"success": True})
        return

    region = config.get("region") or os.getenv("AWS_REGION", DEFAULT_REGION)
    inference_config = config.get("inferenceConfig") or {}
    turn_detection_config = config.get("turnDetectionConfig") or {}

    session = NovaSonicSession(
        session_id=sid,
        region=region,
        model_id=os.getenv("NOVA_SONIC_MODEL_ID", DEFAULT_MODEL_ID),
        inference_config={
            "maxTokens": inference_config.get("maxTokens", 2048),
            "topP": inference_config.get("topP", 0.9),
            "temperature": inference_config.get("temperature", 1),
        },
        turn_detection_config=turn_detection_config,
        on_event=lambda event_name, payload: emit_to_socket(sid, event_name, payload),
    )
    sessions[sid] = session
    session_configs[sid] = config

    await send_ack(sid, message_id, {"success": True})


async def handle_prompt_start(sid: str, data: dict | None) -> None:
    session = sessions.get(sid)
    if not session:
        await emit_to_socket(sid, "error", {"message": "No active session for prompt start"})
        return

    data = data or {}
    try:
        await session.setup_prompt_start(
            voice_id=data.get("voiceId") or "tiffany",
            output_sample_rate=data.get("outputSampleRate") or 24000,
        )
    except Exception as exc:
        await emit_to_socket(
            sid,
            "error",
            {"message": "Error processing prompt start", "details": str(exc)},
        )


async def handle_system_prompt(sid: str, data: str | dict | None) -> None:
    session = sessions.get(sid)
    if not session:
        await emit_to_socket(sid, "error", {"message": "No active session for system prompt"})
        return

    content = data if isinstance(data, str) else (data or {}).get("content", "")
    try:
        await session.send_system_prompt(content)
    except Exception as exc:
        await emit_to_socket(
            sid,
            "error",
            {"message": "Error processing system prompt", "details": str(exc)},
        )


async def handle_audio_start(sid: str) -> None:
    session = sessions.get(sid)
    if not session:
        await emit_to_socket(sid, "error", {"message": "No active session for audio start"})
        return

    try:
        await session.start_audio()
        await emit_to_socket(sid, "audioReady")
    except Exception as exc:
        await emit_to_socket(
            sid,
            "error",
            {"message": "Error processing audio start", "details": str(exc)},
        )


async def handle_audio_input(sid: str, audio_data: str) -> None:
    session = sessions.get(sid)
    if not session:
        await emit_to_socket(sid, "error", {"message": "No active session for audio input"})
        return

    try:
        await session.send_audio_input(audio_data)
    except Exception as exc:
        await emit_to_socket(
            sid,
            "error",
            {"message": "Error processing audio", "details": str(exc)},
        )


async def handle_text_input(sid: str, data: dict | None) -> None:
    session = sessions.get(sid)
    if not session:
        await emit_to_socket(sid, "error", {"message": "No active session for text input"})
        return

    try:
        await session.send_text_input((data or {}).get("content", ""))
    except Exception as exc:
        await emit_to_socket(
            sid,
            "error",
            {"message": "Error processing text input", "details": str(exc)},
        )


async def handle_event(sid: str, message: dict[str, Any]) -> None:
    event = message.get("event")
    data = message.get("data")
    message_id = message.get("id")

    if event == "initializeConnection":
        await handle_initialize_connection(sid, data, message_id)
    elif event == "promptStart":
        await handle_prompt_start(sid, data)
    elif event == "systemPrompt":
        await handle_system_prompt(sid, data)
    elif event == "audioStart":
        await handle_audio_start(sid)
    elif event == "audioInput":
        await handle_audio_input(sid, data)
    elif event == "textInput":
        await handle_text_input(sid, data)
    elif event == "stopAudio":
        await close_session(sid)
        await emit_to_socket(sid, "sessionClosed")
    elif event == "startNewChat":
        await close_session(sid)
        await handle_initialize_connection(sid, data or {}, message_id)
    else:
        await emit_to_socket(sid, "error", {"message": f"Unknown event: {event}"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sid = str(uuid.uuid4())
    connections[sid] = websocket
    connection_locks[sid] = asyncio.Lock()

    print(f"WebSocket connected: {sid}", flush=True)
    await emit_to_socket(sid, "connect")

    try:
        while True:
            message = await websocket.receive_json()
            await handle_event(sid, message)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {sid}", flush=True)
    except Exception as exc:
        if sid in connections:
            await emit_to_socket(sid, "error", {"message": "WebSocket error", "details": str(exc)})
    finally:
        connections.pop(sid, None)
        connection_locks.pop(sid, None)
        await close_session(sid)


app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    host = os.getenv("HOST", "localhost")
    print(f"Python backend listening on http://{host}:{port}", flush=True)
    uvicorn.run("main:app", host=host, port=port, reload=False, app_dir=str(BACKEND_DIR))
