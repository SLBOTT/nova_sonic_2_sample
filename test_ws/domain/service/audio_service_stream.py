import asyncio
from fastapi import WebSocket
from app.domain.services.ai_stream_service import AIStreamService

class AudioStreamService:
    def __init__(self):
        self.ai_service = AIStreamService()

    async def handle_connection(self, websocket: WebSocket):
        ai_stream = await self.ai_service.start_stream()

        async def receive_audio():
            while True:
                audio_chunk = await websocket.receive_bytes()
                await ai_stream.send_audio(audio_chunk)

        async def send_audio():
            while True:
                response_chunk = await ai_stream.receive_audio()
                await websocket.send_bytes(response_chunk)

        await asyncio.gather(
            receive_audio(),
            send_audio()
        )