import asyncio
import os
from pathlib import Path

import socketio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nova_client import DEFAULT_MODEL_ID, NovaSonicSession


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
PUBLIC_DIR = ROOT_DIR / "public"
SOCKET_IO_CLIENT_JS = (
    ROOT_DIR / "node_modules" / "socket.io" / "client-dist" / "socket.io.js"
)
DEFAULT_REGION = "us-east-1"

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(ROOT_DIR / ".env")

api = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(sio, other_asgi_app=api)

sessions: dict[str, NovaSonicSession] = {}
session_configs: dict[str, dict] = {}


@api.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")


@api.get("/api/tools")
async def tools():
    return {"tools": []}


@api.get("/socket.io-client/socket.io.js")
async def socket_io_client():
    return FileResponse(SOCKET_IO_CLIENT_JS)


@api.get("/health")
async def health():
    return {
        "status": "ok",
        "activeSessions": len(sessions),
        "modelId": os.getenv("NOVA_SONIC_MODEL_ID", DEFAULT_MODEL_ID),
    }


api.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")


async def emit_to_socket(sid: str, event_name: str, data: dict) -> None:
    await sio.emit(event_name, data, to=sid)


async def close_session(sid: str) -> None:
    session = sessions.pop(sid, None)
    session_configs.pop(sid, None)
    if session:
        await session.close()


@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}", flush=True)


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}", flush=True)
    await close_session(sid)


@sio.event
async def initializeConnection(sid, data=None):
    config = data or {}
    if sid in sessions:
        return {"success": True}

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

    return {"success": True}


@sio.event
async def promptStart(sid, data=None):
    session = sessions.get(sid)
    if not session:
        await sio.emit("error", {"message": "No active session for prompt start"}, to=sid)
        return

    data = data or {}
    try:
        await session.setup_prompt_start(
            voice_id=data.get("voiceId") or "tiffany",
            output_sample_rate=data.get("outputSampleRate") or 24000,
        )
    except Exception as exc:
        await sio.emit(
            "error",
            {"message": "Error processing prompt start", "details": str(exc)},
            to=sid,
        )


@sio.event
async def systemPrompt(sid, data=None):
    session = sessions.get(sid)
    if not session:
        await sio.emit("error", {"message": "No active session for system prompt"}, to=sid)
        return

    if isinstance(data, str):
        content = data
    else:
        content = (data or {}).get("content", "")

    try:
        await session.send_system_prompt(content)
    except Exception as exc:
        await sio.emit(
            "error",
            {"message": "Error processing system prompt", "details": str(exc)},
            to=sid,
        )


@sio.event
async def audioStart(sid):
    session = sessions.get(sid)
    if not session:
        await sio.emit("error", {"message": "No active session for audio start"}, to=sid)
        return

    try:
        await session.start_audio()
        await sio.emit("audioReady", to=sid)
    except Exception as exc:
        await sio.emit(
            "error",
            {"message": "Error processing audio start", "details": str(exc)},
            to=sid,
        )


@sio.event
async def audioInput(sid, audio_data):
    session = sessions.get(sid)
    if not session:
        await sio.emit("error", {"message": "No active session for audio input"}, to=sid)
        return

    try:
        await session.send_audio_input(audio_data)
    except Exception as exc:
        await sio.emit(
            "error",
            {"message": "Error processing audio", "details": str(exc)},
            to=sid,
        )


@sio.event
async def textInput(sid, data=None):
    session = sessions.get(sid)
    if not session:
        await sio.emit("error", {"message": "No active session for text input"}, to=sid)
        return

    try:
        await session.send_text_input((data or {}).get("content", ""))
    except Exception as exc:
        await sio.emit(
            "error",
            {"message": "Error processing text input", "details": str(exc)},
            to=sid,
        )


@sio.event
async def stopAudio(sid):
    await close_session(sid)
    await sio.emit("sessionClosed", to=sid)


@sio.event
async def startNewChat(sid, data=None):
    await close_session(sid)
    await initializeConnection(sid, data or {})


async def shutdown() -> None:
    await asyncio.gather(*(close_session(sid) for sid in list(sessions.keys())))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    host = os.getenv("HOST", "localhost")
    print(f"Python backend listening on http://{host}:{port}", flush=True)
    uvicorn.run("main:app", host=host, port=port, reload=False, app_dir=str(BACKEND_DIR))
