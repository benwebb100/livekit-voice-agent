#!/usr/bin/env python3
"""
LiveKit Voice Assistant Agent
A multilingual voice assistant using LiveKit Agents with Deepgram STT, OpenAI GPT-4o-mini, and ElevenLabs TTS.
"""

import asyncio
import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc, api
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    ChatContext,
)
from livekit.plugins import openai, deepgram, elevenlabs, silero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()



def test_openai_connection():
    """Test OpenAI API connection."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    
    print(f"OpenAI API Key (first 10 chars): {api_key[:10]}...")
    
    # Test the connection
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("OpenAI API connection successful!")
    except Exception as e:
        print(f"OpenAI API error: {e}")


class MelbourneFitnessAgent(Agent):
    """Melbourne Fitness Studio voice assistant agent."""
    
    def __init__(self):
        # System prompt for the assistant
        system_prompt = """[Identity]  
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
        
        super().__init__(instructions=system_prompt)
    
    async def on_enter(self) -> None:
        """Called when the agent becomes active."""
        await self.session.say("Hi, this is Sarah from Melbourne Fitness Studio. How's it going?")


# async def entrypoint(ctx: JobContext):
#     """Main entry point for the agent."""
    
#     # Create the agent session with all components
#     session = AgentSession(
#         stt=deepgram.STT(
#             api_key=os.getenv("DEEPGRAM_API_KEY"),
#             model="nova-2",
#             language="en-US"
#         ),
#         llm=openai.LLM(
#             api_key=os.getenv("OPENAI_API_KEY"),
#             model="gpt-4o-mini",
#             temperature=0.7
#         ),
#         tts=elevenlabs.TTS(
#             api_key=os.getenv("ELEVENLABS_API_KEY"),
#             voice_id="aGkVQvWUZi16EH8aZJvT",  # Use voice_id instead of voice
#             model="eleven_monolingual_v1"
#         ),
#         vad=silero.VAD.load(),
#     )
    
#     # Start the session with our agent
#     await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
    
#     # Connect to the room
#     await ctx.connect()
    
#     # Keep the agent running
#     await asyncio.Event().wait()

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent."""
    
    # Check if this is a phone call room
    room_name = ctx.room.name
    is_phone_call = room_name.startswith("call-")
    
    # Parse room metadata
    phone_number = None
    if ctx.room.metadata:
        try:
            import json
            metadata = json.loads(ctx.room.metadata)
            is_phone_call = metadata.get("type") == "phone_call"
            phone_number = metadata.get("from")
            logger.info(f"Phone call from: {phone_number}")
        except:
            pass

    # Connect to room first
    await ctx.connect()
    logger.info(f"Connected to room: {room_name}")

    try:
        # Create phone-optimized session
        if is_phone_call:
            await asyncio.sleep(1)
            session = AgentSession(
                stt=deepgram.STT(
                    api_key=os.getenv("DEEPGRAM_API_KEY"),
                    model="nova-2-phonecall",  # Phone-optimized model
                    language="en-US",
                    # Phone-specific settings
                    smart_format=True,
                    punctuate=True,
                    endpointing_ms=1000
                ),
                llm=openai.LLM(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-4o-mini",
                    temperature=0.7
                ),
                tts=elevenlabs.TTS(
                    api_key=os.getenv("ELEVENLABS_API_KEY"),
                    voice_id="aGkVQvWUZi16EH8aZJvT",
                    model="eleven_turbo_v2",  # Low latency model
                    streaming_latency=3,  # Optimize for phone
                ),
                vad=silero.VAD.load(),
            )
            
            logger.info(f"Starting phone session in room: {room_name}")
        else:
            # Regular web session config
            session = AgentSession(
                stt=deepgram.STT(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2",
                language="en-US"
            ),
            llm=openai.LLM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                temperature=0.7
            ),
            tts=elevenlabs.TTS(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id="aGkVQvWUZi16EH8aZJvT",  # Use voice_id instead of voice
                model="eleven_monolingual_v1"
            ),
            vad=silero.VAD.load(),
            )
        
        # Start the session
        await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
    except Exception as e:
        logger.error(f"Failed to start agent session: {e}")
        await ctx.room.close()


async def console_mode():
    """Console mode for testing without LiveKit connection."""
    print("Melbourne Fitness Studio - Console Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    # Create LLM for console mode
    llm = openai.LLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create agent
    agent = MelbourneFitnessAgent()
    
    # Create chat context with system prompt
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content=agent.instructions)
    
    # Initial greeting
    print("Assistant: Hi, this is Sarah from Melbourne Fitness Studio. How's it going?")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Assistant: Thanks for chatting! Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Add user message
            chat_ctx.add_message(role="user", content=user_input)
            
            # Get LLM response
            response_text = ""
            stream = llm.chat(chat_ctx=chat_ctx)
            
            print("Assistant: ", end="", flush=True)
            async for chunk in stream:
                if hasattr(chunk, 'delta') and chunk.delta is not None:
                    if hasattr(chunk.delta, 'content') and chunk.delta.content:
                        print(chunk.delta.content, end="", flush=True)
                        response_text += chunk.delta.content
            print()  # New line after response
            
            # Add assistant response to context
            chat_ctx.add_message(role="assistant", content=response_text)
            
        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py [console|dev|start|download-files]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Validate required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY", 
        "ELEVENLABS_API_KEY",
    ]
    
    # Only require LiveKit credentials for dev/start modes
    if mode in ["dev", "start"]:
        required_vars.extend([
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET"
        ])
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        sys.exit(1)
    
    if mode == "console":
        # Run in console mode
        asyncio.run(console_mode())
    
    elif mode in ["dev", "start", "download-files"]:
        # Use LiveKit CLI to run the agent
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            ws_url=os.getenv("LIVEKIT_URL"),
        )
        
        # Pass the mode to CLI
        cli.run_app(worker_options)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: console, dev, start, download-files")
        sys.exit(1)


if __name__ == "__main__":
    main()
    test_openai_connection()