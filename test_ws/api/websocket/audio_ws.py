from fastapi import WebSocket, WebSocketDisconnect
from test_ws.domain.services.audio_stream_service import AudioStreamService

async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    service = AudioStreamService()

    try:
        await service.handle_connection(websocket)

    except WebSocketDisconnect:
        print("❌ Client disconnected")