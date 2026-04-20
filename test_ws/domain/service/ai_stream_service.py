class AIStreamService:
    async def start_stream(self):
        # Initialize connection to AI (Bedrock, etc.)
        return AIStreamConnection()


class AIStreamConnection:
    async def send_audio(self, chunk: bytes):
        # Send audio to AI model
        pass

    async def receive_audio(self) -> bytes:
        # Receive audio from AI model
        return b""  # replace with real stream