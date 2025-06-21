#!/usr/bin/env python3
"""
FastAPI Twilio Integration for LiveKit Voice Assistant
Handles incoming/outgoing phone calls and connects them to the LiveKit agent.
"""

import os
import logging
import asyncio
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from livekit import api
from contextlib import asynccontextmanager

# Load environment variables
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

# Global variable for LiveKit API client
livekit_api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize and cleanup resources."""
    global livekit_api
    
    # Startup
    logger.info("Starting up FastAPI application...")
    
    # Initialize LiveKit API client in async context
    livekit_api = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    if livekit_api and hasattr(livekit_api, '_session'):
        await livekit_api._session.close()


# Initialize FastAPI app with lifespan
app = FastAPI(title="LiveKit Twilio Integration", lifespan=lifespan)

# Initialize Twilio client (sync client is fine)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


class OutboundCallRequest(BaseModel):
    to: str
    metadata: Optional[dict] = None


def generate_livekit_token(room_name: str, participant_name: str) -> str:
    """Generate a LiveKit access token for the phone participant."""
    token = api.AccessToken(
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    )
    token.add_grant(
        api.VideoGrant(
            room_join=True,
            room=room_name
        )
    )
    token.identity = participant_name
    token.metadata = "phone_participant"
    
    return token.to_jwt()


@app.post("/twilio/voice")
async def handle_incoming_call(request: Request):
    """
    Handle incoming calls from Twilio and connect them to LiveKit.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From")
    to_number = form_data.get("To")
    
    logger.info(f"Incoming call from {from_number} to {to_number}")
    
    # Create a unique room name for this call
    room_name = f"call-{call_sid}"
    
    # Create the room in LiveKit
    try:
        await livekit_api.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                metadata=f"phone_call:{from_number}"
            )
        )
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
    
    # Generate TwiML response to connect the call to LiveKit
    response = VoiceResponse()
    
    # Initial greeting while connecting
    response.say("Connecting you to Melbourne Fitness Studio.", voice="Polly.Joanna")
    
    # For now, we'll use a simpler approach with webhook
    # LiveKit's Twilio integration typically requires SIP
    # This is a placeholder - you'll need to set up proper SIP integration
    
    # Temporary: Just use Twilio's built-in TTS for demo
    response.say("Hi, this is Sarah from Melbourne Fitness Studio. How's it going?", voice="Polly.Joanna")
    response.pause(length=2)
    response.say("Our AI assistant is currently being connected. Please call back in a moment.", voice="Polly.Joanna")
    response.hangup()
    
    return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/twilio/outbound")
async def initiate_outbound_call(request: OutboundCallRequest):
    """
    Initiate an outbound call from the Twilio number to a customer.
    """
    try:
        # Get the base URL from request
        base_url = str(request.url).replace('/twilio/outbound', '')
        
        # Create a unique room name for this call
        call_id = f"outbound-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        room_name = f"call-{call_id}"
        
        # Create the room in LiveKit
        await livekit_api.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                metadata=f"outbound_call:{request.to}"
            )
        )
        
        # Make the call
        call = twilio_client.calls.create(
            to=request.to,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{base_url}/twilio/voice",
            status_callback=f"{base_url}/twilio/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )
        
        return {
            "status": "initiated",
            "sid": call.sid,
            "room_name": room_name,
            "to": request.to
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/twilio/status")
async def handle_call_status(request: Request):
    """
    Handle call status updates from Twilio.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # Handle call completion
    if call_status == "completed":
        # Clean up the LiveKit room
        room_name = f"call-{call_sid}"
        try:
            await livekit_api.room.delete_room(
                api.DeleteRoomRequest(room=room_name)
            )
            logger.info(f"Deleted room {room_name}")
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
    
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "twilio_configured": bool(TWILIO_ACCOUNT_SID),
        "livekit_configured": bool(LIVEKIT_URL),
        "api_initialized": livekit_api is not None
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LiveKit Twilio Integration",
        "endpoints": [
            "/health",
            "/twilio/voice",
            "/twilio/outbound",
            "/twilio/status"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Verify required environment variables
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
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )