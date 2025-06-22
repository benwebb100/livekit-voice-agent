#!/usr/bin/env python3
"""
Direct Twilio Media Streams Voice Agent
Connects Twilio directly to AI services without LiveKit
"""

import os
import json
import base64
import asyncio
import logging
import audioop
from typing import Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Say, Stream
from pydantic import BaseModel
import numpy as np
import openai
from deepgram import DeepgramClient, PrerecordedOptions, LiveTranscriptionEvents
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
import httpx
import io

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Global variables
active_calls: Dict[str, 'CallSession'] = {}

# System prompt
SYSTEM_PROMPT = """[Identity]  
You are a friendly and articulate AI phone agent working for Melbourne Fitness Studio. You are calling a real customer who has already booked a tour of the studio. Your role is to simulate how a real front desk receptionist would handle a welcome and info-gathering call before their scheduled tour. You are not an AI demo assistant. This is not a demo scenario — act like a human staff member confirming the booking and collecting information to help personalize the visit.

[Style]  
- Speak clearly and naturally.  
- Use a warm, conversational tone.  
- Avoid robotic or overly formal speech.  
- Use casual phrasing and human-like timing (e.g., "Alright…", "Cool!").  
- Keep sentences short and natural.  
- Reflect the tone of the business: supportive, friendly, energetic

[Response Guidelines]  
- Stay in character as a real staff member from Melbourne Fitness Studio.  
- Always ask one question at a time and wait for a response before moving to the next.  
- Do not rush the call — you are building rapport and excitement.  
- If the user needs to reschedule, acknowledge politely and suggest someone from the team will follow up.  
- Never say you're an AI.  
- Do not offer real-time bookings or calendar integration — this is a simulation only.  
- End the call once all core information is collected and the tour is reconfirmed."""


class CallSession:
    """Manages a single phone call session"""
    
    def __init__(self, call_sid: str, stream_sid: str, websocket: WebSocket):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.websocket = websocket
        self.from_number = ""
        self.to_number = ""
        
        # Audio buffers
        self.audio_buffer = bytearray()
        self.response_queue = asyncio.Queue()
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi, this is Sarah from Melbourne Fitness Studio. How's it going?"}
        ]
        
        # Control flags
        self.is_processing = False
        self.call_active = True
        
        # Deepgram live transcription
        self.deepgram_live = None
        self.transcript_buffer = ""
        self.silence_start = None
        
    async def start(self):
        """Initialize the session and send greeting"""
        # Generate and send initial greeting
        await self.say_text("Hi, this is Sarah from Melbourne Fitness Studio. How's it going?")
        
        # Start Deepgram live transcription
        await self.start_deepgram()
        
    async def start_deepgram(self):
        """Start Deepgram live transcription"""
        try:
            from deepgram import LiveTranscriptionEvents, LiveOptions

            options = LiveOptions(
            model="nova-2-phonecall",
            language="en-US",
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing=300,
            vad_events=True
            )

            # self.deepgram_live = deepgram_client.listen.live.v("1")
            self.deepgram_live = deepgram_client.listen.websocket.v("1")
            # self.deepgram_live = await deepgram_client.transcription.live({
            #     'model': 'nova-2-phonecall',
            #     'language': 'en-US',
            #     'smart_format': True,
            #     'punctuate': True,
            #     'interim_results': True,
            #     'endpointing': 300,
            #     'vad_events': True
            # })
            
            # # Set up event handlers
            # self.deepgram_live.register_handler(
            #     LiveTranscriptionEvents.Transcript,
            #     self.on_deepgram_transcript
            # )
            
            # self.deepgram_live.register_handler(
            #     LiveTranscriptionEvents.SpeechStarted,
            #     self.on_speech_started
            # )
            # Set up event handlers

             # Set up event handlers with the correct syntax
            self.deepgram_live.on("transcript", self.on_deepgram_transcript)
            self.deepgram_live.on("speech_started", self.on_speech_started)
            
            # Start the connection
            await self.deepgram_live.start(options)
            logger.info(f"Deepgram live transcription started for {self.call_sid}")
            
        except Exception as e:
            logger.error(f"Failed to start Deepgram: {e}")
            import traceback
            traceback.print_exc()
    
    # async def on_deepgram_transcript(self, result):
    #     """Handle Deepgram transcription results"""
    #     if not result.is_final:
    #         return
            
    #     transcript = result.channel.alternatives[0].transcript.strip()
    #     if transcript:
    #         logger.info(f"User said: {transcript}")
    #         self.transcript_buffer = transcript
            
    #         # Process after a short delay to catch multi-sentence inputs
    #         await asyncio.sleep(0.5)
    #         if self.transcript_buffer == transcript:  # No new input
    #             await self.process_user_input(transcript)
    #             self.transcript_buffer = ""
    def on_deepgram_transcript(self, *args, **kwargs):
        """Handle Deepgram transcription results"""
        result = kwargs.get("result", {})
        if not result:
            return
            
        channel = result.get("channel", {})
        alternatives = channel.get("alternatives", [])
        
        if alternatives and result.get("is_final"):
            transcript = alternatives[0].get("transcript", "").strip()
            if transcript:
                logger.info(f"User said: {transcript}")
                asyncio.create_task(self.process_user_input(transcript))

    def on_speech_started(self, *args, **kwargs):
        """Handle when user starts speaking"""
        logger.info("Speech started")
    
    async def on_speech_started(self, *args):
        """Handle when user starts speaking"""
        # Could implement interruption logic here
        pass
    
    async def process_user_input(self, text: str):
        """Process user input and generate response"""
        if self.is_processing or not text:
            return
            
        self.is_processing = True
        
        try:
            # Add user message to conversation
            self.messages.append({"role": "user", "content": text})
            
            # Generate response using OpenAI
            response = await self.generate_llm_response()
            
            if response:
                # Add assistant response to conversation
                self.messages.append({"role": "assistant", "content": response})
                
                # Convert response to speech and send
                await self.say_text(response)
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
        finally:
            self.is_processing = False
    
    async def generate_llm_response(self) -> str:
        """Generate response using OpenAI"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                temperature=0.7,
                max_tokens=150,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None
    
    async def say_text(self, text: str):
        """Convert text to speech using ElevenLabs and send to Twilio"""
        try:
            # Use ElevenLabs Python SDK with proper method
            voice_id = "aGkVQvWUZi16EH8aZJvT"  # Sarah voice
            
            # Convert text to speech using the SDK
            audio_generator = elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2",
                output_format="ulaw_8000",  # Direct ulaw format for Twilio
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75
                )
            )
            
            # Collect audio chunks
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)
            
            # Combine all chunks
            audio_data = b''.join(audio_chunks)
            
            # Send audio in chunks to Twilio
            chunk_size = 160  # 20ms at 8kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                
                # Send to Twilio
                await self.send_audio_to_twilio(chunk)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.02)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio data to Twilio"""
        if not self.websocket or not self.call_active:
            return
            
        try:
            # Encode to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Create media message
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": audio_base64
                }
            }
            
            await self.websocket.send_text(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending audio to Twilio: {e}")
    
    # async def handle_audio_from_twilio(self, audio_payload: str):
    #     """Handle incoming audio from Twilio"""
    #     try:
    #         # Decode base64 μ-law audio
    #         ulaw_audio = base64.b64decode(audio_payload)
            
    #         # Convert μ-law to PCM for Deepgram
    #         pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
            
    #         # Send to Deepgram for transcription
    #         if self.deepgram_live:
    #             await self.deepgram_live.send(pcm_audio)
                
    #     except Exception as e:
    #         logger.error(f"Error handling audio: {e}")

    async def handle_audio_from_twilio(self, audio_payload: str):
        """Handle incoming audio from Twilio"""
        try:
            # Decode base64 μ-law audio
            ulaw_audio = base64.b64decode(audio_payload)
            
            # Convert μ-law to PCM for Deepgram
            pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
            
            # Send to Deepgram for transcription
            if self.deepgram_live:
                self.deepgram_live.send(pcm_audio)
                    
        except Exception as e:
            logger.error(f"Error handling audio: {e}")
            
    async def cleanup(self):
        """Clean up session resources"""
        self.call_active = False
        
        if self.deepgram_live:
            try:
                await self.deepgram_live.finish()
            except:
                pass
        
        logger.info(f"Cleaned up session for {self.call_sid}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Direct Twilio Voice Agent...")
    yield
    logger.info("Shutting down...")
    
    # Clean up active calls
    for session in active_calls.values():
        await session.cleanup()


app = FastAPI(title="Direct Twilio Voice Agent", lifespan=lifespan)


class OutboundCallRequest(BaseModel):
    to: str


@app.post("/twilio/voice")
async def handle_voice(request: Request):
    """Handle incoming voice call and set up media stream"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From", "Unknown")
    to_number = form_data.get("To", "Unknown")
    
    logger.info(f"Incoming call {call_sid} from {from_number} to {to_number}")
    
    # Generate TwiML response
    response = VoiceResponse()
    
    # Brief greeting while connecting
    response.say("Connecting your call.", voice="Polly.Joanna")
    
    # Start media stream
    start = Start()
    
    # Get base URL and convert to WebSocket URL
    base_url = str(request.url).replace('/twilio/voice', '')
    ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
    
    # Configure stream
    stream = start.stream(
        url=f"{ws_url}/twilio/media-stream",
        name=call_sid
    )
    
    # Pass metadata
    stream.parameter(name="callSid", value=call_sid)
    stream.parameter(name="fromNumber", value=from_number)
    stream.parameter(name="toNumber", value=to_number)
    
    response.append(start)
    
    # Keep call alive
    response.pause(length=3600)
    
    return PlainTextResponse(str(response), media_type="application/xml")


@app.websocket("/twilio/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle Twilio media stream WebSocket connection"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    session = None
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "start":
                # Extract stream information
                stream_sid = data["streamSid"]
                call_sid = data["start"]["callSid"]
                
                # Get custom parameters
                custom_params = data["start"]["customParameters"]
                
                # Create new session
                session = CallSession(call_sid, stream_sid, websocket)
                session.from_number = custom_params.get("fromNumber", "Unknown")
                session.to_number = custom_params.get("toNumber", "Unknown")
                
                # Store session
                active_calls[call_sid] = session
                
                logger.info(f"Stream started - Call: {call_sid}, Stream: {stream_sid}")
                
                # Start the session
                await session.start()
                
            elif event_type == "media" and session:
                # Handle incoming audio
                payload = data["media"]["payload"]
                await session.handle_audio_from_twilio(payload)
                
            elif event_type == "stop":
                logger.info(f"Media stream stopped")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up session
        if session:
            await session.cleanup()
            if session.call_sid in active_calls:
                del active_calls[session.call_sid]


@app.post("/twilio/outbound")
async def initiate_outbound_call(request: Request, call_request: OutboundCallRequest):
    """Initiate an outbound call"""
    try:
        # Get base URL
        base_url = str(request.url).replace('/twilio/outbound', '')
        
        # Make the call
        call = twilio_client.calls.create(
            to=call_request.to,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{base_url}/twilio/voice",
            status_callback=f"{base_url}/twilio/status"
        )
        
        logger.info(f"Initiated outbound call: {call.sid} to {call_request.to}")
        
        return {
            "status": "initiated",
            "call_sid": call.sid,
            "to": call_request.to
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/twilio/status")
async def handle_call_status(request: Request):
    """Handle call status callbacks"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # Clean up session if call ended
    if call_status in ["completed", "failed", "busy", "no-answer"]:
        if call_sid in active_calls:
            session = active_calls[call_sid]
            await session.cleanup()
            del active_calls[call_sid]
    
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_calls": len(active_calls),
        "twilio_configured": bool(TWILIO_ACCOUNT_SID),
        "openai_configured": bool(OPENAI_API_KEY),
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY)
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Direct Twilio Voice Agent",
        "endpoints": {
            "health": "/health",
            "incoming_calls": "/twilio/voice",
            "outbound_calls": "/twilio/outbound",
            "media_stream": "/twilio/media-stream",
            "call_status": "/twilio/status"
        },
        "active_calls": len(active_calls)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Verify environment variables
    required_vars = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN", 
        "TWILIO_PHONE_NUMBER",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )