#!/usr/bin/env python3
"""
FastAPI Twilio Media Streams Integration for LiveKit Voice Assistant
Complete working implementation
"""

import os
import json
import base64
import asyncio
import logging
import audioop
import struct
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, Request, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Say, Stream
from livekit import api, rtc
from pydantic import BaseModel
import numpy as np

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Global variables
livekit_api = None
active_streams: Dict[str, dict] = {}
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# Add this near the top of your file after imports
def test_token_generation():
    """Test which token generation method works."""
    try:
        # Test method 1
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        if hasattr(token, 'add_grant'):
            print("Method 1: add_grant exists")
        elif hasattr(token, 'with_grants'):
            print("Method 2: with_grants exists")
        else:
            print("Available methods:", dir(token))
    except Exception as e:
        print(f"Error testing token: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global livekit_api
    
    # Startup
    logger.info("Starting FastAPI application...")
    livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if livekit_api and hasattr(livekit_api, '_session'):
        await livekit_api._session.close()


app = FastAPI(title="LiveKit Twilio Media Streams", lifespan=lifespan)


class OutboundCallRequest(BaseModel):
    to: str
    
    
class StreamInfo:
    def __init__(self):
        self.room_name: Optional[str] = None
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.room: Optional[rtc.Room] = None
        self.audio_source: Optional[rtc.AudioSource] = None
        self.audio_track: Optional[rtc.LocalAudioTrack] = None
        self.from_number: Optional[str] = None
        self.to_number: Optional[str] = None
        # For receiving audio from LiveKit
        self.subscribed_track: Optional[rtc.RemoteAudioTrack] = None
        self.websocket: Optional[WebSocket] = None


# def generate_token(room_name: str, identity: str) -> str:
#     """Generate LiveKit access token."""
#     token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token.add_grant(api.VideoGrant(
#         room_join=True,
#         room=room_name,
#         can_publish=True,
#         can_subscribe=True
#     ))
#     token.identity = identity
#     token.metadata = "phone_participant"
#     return token.to_jwt()

from livekit import api

def generate_token(room_name: str, identity: str) -> str:
    """Generate LiveKit access token."""
    # Create the token with API credentials
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    # Set identity and metadata
    token.identity = identity
    token.name = identity
    token.metadata = "phone_participant"
    
    # Add room permissions
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True
    ))
    
    return token.to_jwt()


def ulaw_decode(ulaw_bytes: bytes) -> bytes:
    """Decode μ-law audio to 16-bit PCM."""
    # Use Python's audioop for proper μ-law decoding
    # Convert from 8-bit μ-law to 16-bit linear PCM
    pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
    return pcm_data


def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Encode 16-bit PCM to μ-law."""
    # Convert from 16-bit linear PCM to 8-bit μ-law
    ulaw_data = audioop.lin2ulaw(pcm_data, 2)
    return ulaw_data


@app.post("/twilio/voice")
async def handle_voice(request: Request):
    """Handle incoming voice call and set up media stream."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From", "Unknown")
    to_number = form_data.get("To", "Unknown")
    
    logger.info(f"Incoming call {call_sid} from {from_number} to {to_number}")
    
    # Create room name
    room_name = f"call-{call_sid}"
    
    # Create LiveKit room with metadata
    try:
        await livekit_api.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                metadata=json.dumps({
                    "type": "phone_call",
                    "from": from_number,
                    "to": to_number,
                    "call_sid": call_sid
                })
            )
        )
        logger.info(f"Created LiveKit room: {room_name}")
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
    
    # Generate TwiML response
    response = VoiceResponse()
    
    # Brief greeting
    response.say("Connecting your call.", voice="Polly.Joanna")
    
    # Start media stream
    start = Start()
    
    # Get base URL and convert to WebSocket URL
    base_url = str(request.url).replace('/twilio/voice', '')
    ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
    
    # Configure stream with custom parameters
    stream = start.stream(
        url=f"{ws_url}/twilio/media-stream",
        name=room_name
    )
    # Pass metadata through custom parameters
    stream.parameter(name="roomName", value=room_name)
    stream.parameter(name="fromNumber", value=from_number)
    stream.parameter(name="toNumber", value=to_number)
    
    response.append(start)
    
    # Keep call alive with pause
    response.pause(length=3600)  # 1 hour max
    
    return PlainTextResponse(str(response), media_type="application/xml")


@app.websocket("/twilio/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle Twilio media stream WebSocket connection."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    stream_info = StreamInfo()
    stream_info.websocket = websocket
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "start":
                await handle_stream_start(data, stream_info)
                
            elif event_type == "media":
                await handle_stream_media(data, stream_info)
                
            elif event_type == "stop":
                logger.info(f"Media stream stopped: {stream_info.stream_sid}")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await cleanup_stream(stream_info)


async def handle_stream_start(data: dict, stream_info: StreamInfo):
    """Handle stream start event."""
    # Extract stream information
    stream_info.stream_sid = data["streamSid"]
    stream_info.call_sid = data["start"]["callSid"]
    
    # Get custom parameters
    custom_params = data["start"]["customParameters"]
    stream_info.room_name = custom_params.get("roomName", f"call-{stream_info.call_sid}")
    stream_info.from_number = custom_params.get("fromNumber", "Unknown")
    stream_info.to_number = custom_params.get("toNumber", "Unknown")
    
    logger.info(f"Stream started - SID: {stream_info.stream_sid}, Room: {stream_info.room_name}")
    
    # Connect to LiveKit room
    await connect_to_livekit(stream_info)
    
    # Store active stream
    active_streams[stream_info.stream_sid] = stream_info


async def connect_to_livekit(stream_info: StreamInfo):
    """Connect to LiveKit room and set up audio tracks."""
    try:
        # Create room instance
        stream_info.room = rtc.Room()
        
        # Set up event handlers
        @stream_info.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, *args):
            """Handle when we receive audio from the agent."""
            if isinstance(track, rtc.RemoteAudioTrack):
                logger.info(f"Subscribed to audio track: {track.sid}")
                stream_info.subscribed_track = track
                # Start forwarding audio to Twilio
                asyncio.create_task(forward_audio_to_twilio(stream_info))
        
        # Create audio source for phone input
        stream_info.audio_source = rtc.AudioSource(
            sample_rate=8000,  # Twilio uses 8kHz
            num_channels=1
        )
        
        # Create audio track
        stream_info.audio_track = rtc.LocalAudioTrack.create_audio_track(
            "phone_input", 
            stream_info.audio_source
        )
        
        # Generate token
        token = generate_token(
            stream_info.room_name, 
            f"phone-{stream_info.from_number}"
        )
        
        # Connect to room
        await stream_info.room.connect(LIVEKIT_URL, token)
        logger.info(f"Connected to LiveKit room: {stream_info.room_name}")
        
        # Publish phone audio track
        await stream_info.room.local_participant.publish_track(
            stream_info.audio_track,
            rtc.TrackPublishOptions(
                name="phone_input",
                source=rtc.TrackSource.SOURCE_MICROPHONE
            )
        )
        logger.info("Published phone audio track")
        
    except Exception as e:
        logger.error(f"Failed to connect to LiveKit: {e}")
        raise


async def handle_stream_media(data: dict, stream_info: StreamInfo):
    """Handle incoming audio from Twilio."""
    if not stream_info.audio_source:
        return
    
    try:
        # Get audio payload
        payload = data["media"]["payload"]
        
        # Decode base64 to get μ-law audio
        ulaw_audio = base64.b64decode(payload)
        
        # Convert μ-law to PCM
        pcm_audio = ulaw_decode(ulaw_audio)
        
        # Convert to numpy array for processing
        audio_array = np.frombuffer(pcm_audio, dtype=np.int16)
        
        # Create audio frame
        frame = rtc.AudioFrame.create(
            sample_rate=8000,
            num_channels=1,
            samples_per_channel=len(audio_array)
        )
        
        # Copy audio data to frame
        np.copyto(
            np.frombuffer(frame.data, dtype=np.int16),
            audio_array
        )
        
        # Send to LiveKit
        await stream_info.audio_source.capture_frame(frame)
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")


async def forward_audio_to_twilio(stream_info: StreamInfo):
    """Forward audio from LiveKit agent to Twilio."""
    if not stream_info.subscribed_track or not stream_info.websocket:
        return
    
    logger.info("Starting audio forwarding to Twilio")
    
    try:
        # Create an audio stream from the track
        audio_stream = rtc.AudioStream(stream_info.subscribed_track)
        
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                frame = event.frame
                
                # Convert audio to proper format for Twilio
                # LiveKit typically uses 48kHz, need to resample to 8kHz
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                # Resample from LiveKit's sample rate to 8kHz for Twilio
                if frame.sample_rate != 8000:
                    # Simple downsampling (for production, use proper resampling)
                    factor = frame.sample_rate // 8000
                    audio_data = audio_data[::factor]
                
                # Convert to bytes
                pcm_bytes = audio_data.tobytes()
                
                # Convert PCM to μ-law
                ulaw_audio = pcm_to_ulaw(pcm_bytes)
                
                # Encode to base64
                audio_base64 = base64.b64encode(ulaw_audio).decode('utf-8')
                
                # Send to Twilio
                media_message = {
                    "event": "media",
                    "streamSid": stream_info.stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                await stream_info.websocket.send_text(json.dumps(media_message))
                
    except Exception as e:
        logger.error(f"Error forwarding audio to Twilio: {e}")


async def cleanup_stream(stream_info: StreamInfo):
    """Clean up resources when stream ends."""
    try:
        # Remove from active streams
        if stream_info.stream_sid and stream_info.stream_sid in active_streams:
            del active_streams[stream_info.stream_sid]
        
        # Disconnect from LiveKit
        if stream_info.room:
            await stream_info.room.disconnect()
            logger.info(f"Disconnected from room: {stream_info.room_name}")
        
        # Delete the room
        if stream_info.room_name and livekit_api:
            try:
                await livekit_api.room.delete_room(
                    api.DeleteRoomRequest(room=stream_info.room_name)
                )
                logger.info(f"Deleted room: {stream_info.room_name}")
            except Exception as e:
                logger.error(f"Failed to delete room: {e}")
                
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


@app.post("/twilio/outbound")
async def initiate_outbound_call(request: Request, call_request: OutboundCallRequest):
    """Initiate an outbound call."""
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
            "sid": call.sid,
            "room_name": f"call-{call.sid}",
            "to": call_request.to
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/twilio/status")
async def handle_call_status(request: Request):
    """Handle call status callbacks."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # No need to handle here as cleanup happens in WebSocket disconnect
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "twilio_configured": bool(TWILIO_ACCOUNT_SID),
        "livekit_configured": bool(LIVEKIT_URL),
        "active_streams": len(active_streams)
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LiveKit Twilio Media Streams Integration",
        "endpoints": {
            "health": "/health",
            "incoming_calls": "/twilio/voice",
            "outbound_calls": "/twilio/outbound",
            "media_stream": "/twilio/media-stream",
            "call_status": "/twilio/status"
        },
        "active_streams": len(active_streams)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Verify environment variables
    required_vars = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN", 
        "TWILIO_PHONE_NUMBER",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
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
    test_token_generation()
    