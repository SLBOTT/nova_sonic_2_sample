from fastapi import FastAPI, WebSocket
from test_ws.api.websocket import audio_websocket_endpoint

app = FastAPI()

@app.websocket("/ws/audio")
async def websocket_route(websocket: WebSocket):
    await audio_websocket_endpoint(websocket)