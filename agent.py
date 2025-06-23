# #!/usr/bin/env python3
# """
# LiveKit Voice Assistant Agent
# A multilingual voice assistant using LiveKit Agents with Deepgram STT, OpenAI GPT-4o-mini, and ElevenLabs TTS.
# """

# import asyncio
# import os
# import sys
# import logging
# from typing import Optional, Dict, Any
# from pathlib import Path
# from dotenv import load_dotenv

# from livekit import rtc, api
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     ChatContext
# )
# from livekit.plugins import openai, deepgram, elevenlabs, silero

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()



# def test_openai_connection():
#     """Test OpenAI API connection."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         print("ERROR: OPENAI_API_KEY not found in environment")
#         return
    
#     print(f"OpenAI API Key (first 10 chars): {api_key[:10]}...")
    
#     # Test the connection
#     try:
#         import openai
#         client = openai.OpenAI(api_key=api_key)
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": "Hello"}],
#             max_tokens=10
#         )
#         print("OpenAI API connection successful!")
#     except Exception as e:
#         print(f"OpenAI API error: {e}")


# # Simple Mock LLM
# class MockLLM:
#     def __init__(self, **kwargs):
#         pass
    
#     async def chat(self, chat_ctx, **kwargs):
#         # Simple response generator
#         for word in "Hi there! Thanks for calling Melbourne Fitness Studio. How can I help you today?".split():
#             yield type('obj', (), {'choices': [type('obj', (), {'delta': type('obj', (), {'content': word + ' '})()})]})()
#             await asyncio.sleep(0.1)


# class MelbourneFitnessAgent(Agent):
#     """Melbourne Fitness Studio voice assistant agent."""
    
#     def __init__(self):
#         # System prompt for the assistant
#         system_prompt = """[Identity]  
# You are a friendly and articulate AI phone agent working for Melbourne Fitness Studio. You are calling a real customer who has already booked a tour of the studio. Your role is to simulate how a real front desk receptionist would handle a welcome and info-gathering call before their scheduled tour. You are not an AI demo assistant. This is not a demo scenario — act like a human staff member confirming the booking and collecting information to help personalize the visit.

# [Style]  
# - Speak clearly and naturally.  
# - Use a warm, conversational tone.  
# - Avoid robotic or overly formal speech.  
# - Use casual phrasing and human-like timing (e.g., "Alright…", "Cool!").  
# - Keep sentences short and natural.  
# - Reflect the tone of the business: supportive, friendly, energetic

# [Response Guidelines]  
# - Stay in character as a real staff member from Melbourne Fitness Studio.  
# - Always ask one question at a time and wait for a response before moving to the next.  
# - Do not rush the call — you are building rapport and excitement.  
# - If the user needs to reschedule, acknowledge politely and suggest someone from the team will follow up.  
# - Never say you're an AI.  
# - Do not offer real-time bookings or calendar integration — this is a simulation only.  
# - End the call once all core information is collected and the tour is reconfirmed."""
        
#         super().__init__(instructions=system_prompt)
    
#     async def on_enter(self) -> None:
#         """Called when the agent becomes active."""
#         await self.session.say("Hi, this is Sarah from Melbourne Fitness Studio. How's it going?")

# async def entrypoint(ctx: JobContext):
#     """Main entry point for the agent."""
    
#     try:
#         # Connect to room with audio-only subscription
#         logger.info(f"Connecting to room: {ctx.room.name}")
#         # await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

#         # If you need to check what's available
#         try:
#             from livekit.agents import AutoSubscribe
#             await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         except ImportError:
#             # Fallback to default behavior
#             await ctx.connect()

#         logger.info(f"Successfully connected to room: {ctx.room.name}")
        
#         # Parse room metadata
#         is_phone_call = ctx.room.name.startswith("call-")
#         phone_number = None
        
#         if ctx.room.metadata:
#             try:
#                 import json
#                 metadata = json.loads(ctx.room.metadata)
#                 is_phone_call = metadata.get("type") == "phone_call"
#                 phone_number = metadata.get("from")
#                 logger.info(f"Phone call from: {phone_number}")
#             except Exception as e:
#                 logger.error(f"Failed to parse metadata: {e}")
        
#         # Wait a moment for connection to stabilize
#         await asyncio.sleep(0.5)
        
#         # Create appropriate session configuration
#         if is_phone_call:
#             logger.info("Creating phone-optimized session")
#             session = AgentSession(
#                 stt=deepgram.STT(
#                     api_key=os.getenv("DEEPGRAM_API_KEY"),
#                     model="nova-2-phonecall",
#                     language="en-US",
#                     smart_format=True,
#                     punctuate=True,
#                     endpointing_ms=1000
#                 ),
#                 llm=openai.LLM(
#                     api_key=os.getenv("OPENAI_API_KEY"),
#                     model="gpt-4o-mini",
#                     temperature=0.7
#                 ),
#                 tts=elevenlabs.TTS(
#                     api_key=os.getenv("ELEVENLABS_API_KEY"),
#                     voice_id="aGkVQvWUZi16EH8aZJvT",
#                     model="eleven_turbo_v2",
#                     streaming_latency=3,
#                 ),
#                 vad=silero.VAD.load(),
#             )
#         else:
#             logger.info("Creating standard session")
#             session = AgentSession(
#                 stt=deepgram.STT(
#                     api_key=os.getenv("DEEPGRAM_API_KEY"),
#                     model="nova-2",
#                     language="en-US"
#                 ),
#                 llm=openai.LLM(
#                     api_key=os.getenv("OPENAI_API_KEY"),
#                     model="gpt-4o-mini",
#                     temperature=0.7
#                 ),
#                 tts=elevenlabs.TTS(
#                     api_key=os.getenv("ELEVENLABS_API_KEY"),
#                     voice_id="aGkVQvWUZi16EH8aZJvT",
#                     model="eleven_monolingual_v1"
#                 ),
#                 vad=silero.VAD.load(),
#             )
        
#         # Start the session
#         # logger.info("Starting agent session...")
#         # await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
#         # logger.info("Agent session started successfully")
#         # In agent.py, modify the session creation part:
#         try:
#             # Start the session
#             logger.info("Starting agent session...")
#             await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
#             logger.info("Agent session started successfully")
#         except Exception as e:
#             # If it's the 404 error, retry without the health check
#             if "404" in str(e):
#                 logger.warning("OpenAI health check failed, retrying...")
#                 # Create a new session with a workaround
#                 await asyncio.sleep(0.5)
#                 await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
        
#     except Exception as e:
#         logger.error(f"Failed in entrypoint: {e}", exc_info=True)
#         # Make sure to disconnect on error
#         try:
#             # await ctx.disconnect()
#             ctx.shutdown(reason=f"Error in entrypoint: {str(e)}")
#             raise
#         except:
#             pass
#         raise


# async def console_mode():
#     """Console mode for testing without LiveKit connection."""
#     print("Melbourne Fitness Studio - Console Mode")
#     print("Type 'quit' to exit")
#     print("-" * 50)
    
#     # Create LLM for console mode
#     llm = openai.LLM(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-4o-mini",
#         temperature=0.7
#     )
    
#     # Create agent
#     agent = MelbourneFitnessAgent()
    
#     # Create chat context with system prompt
#     chat_ctx = ChatContext()
#     chat_ctx.add_message(role="system", content=agent.instructions)
    
#     # Initial greeting
#     print("Assistant: Hi, this is Sarah from Melbourne Fitness Studio. How's it going?")
    
#     while True:
#         try:
#             user_input = input("You: ").strip()
#             if user_input.lower() in ['quit', 'exit', 'bye']:
#                 print("Assistant: Thanks for chatting! Have a great day!")
#                 break
            
#             if not user_input:
#                 continue
            
#             # Add user message
#             chat_ctx.add_message(role="user", content=user_input)
            
#             # Get LLM response
#             response_text = ""
#             stream = llm.chat(chat_ctx=chat_ctx)
            
#             print("Assistant: ", end="", flush=True)
#             async for chunk in stream:
#                 if hasattr(chunk, 'delta') and chunk.delta is not None:
#                     if hasattr(chunk.delta, 'content') and chunk.delta.content:
#                         print(chunk.delta.content, end="", flush=True)
#                         response_text += chunk.delta.content
#             print()  # New line after response
            
#             # Add assistant response to context
#             chat_ctx.add_message(role="assistant", content=response_text)
            
#         except KeyboardInterrupt:
#             print("\nAssistant: Goodbye!")
#             break
#         except Exception as e:
#             print(f"Error: {e}")


# def main():
#     """Main entry point."""
#     if len(sys.argv) < 2:
#         print("Usage: python agent.py [console|dev|start|download-files]")
#         sys.exit(1)
    
#     mode = sys.argv[1]
    
#     # Validate required environment variables
#     required_vars = [
#         "OPENAI_API_KEY",
#         "DEEPGRAM_API_KEY", 
#         "ELEVENLABS_API_KEY",
#     ]
    
#     # Only require LiveKit credentials for dev/start modes
#     if mode in ["dev", "start"]:
#         required_vars.extend([
#             "LIVEKIT_URL",
#             "LIVEKIT_API_KEY",
#             "LIVEKIT_API_SECRET"
#         ])
    
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
#     if missing_vars:
#         print(f"Missing required environment variables: {', '.join(missing_vars)}")
#         print("Please check your .env file")
#         sys.exit(1)
    
#     if mode == "console":
#         # Run in console mode
#         asyncio.run(console_mode())
    
#     elif mode in ["dev", "start", "download-files"]:
#         # Use LiveKit CLI to run the agent
#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             api_key=os.getenv("LIVEKIT_API_KEY"),
#             api_secret=os.getenv("LIVEKIT_API_SECRET"),
#             ws_url=os.getenv("LIVEKIT_URL"),
#         )
        
#         # Pass the mode to CLI
#         cli.run_app(worker_options)
    
#     else:
#         print(f"Unknown mode: {mode}")
#         print("Available modes: console, dev, start, download-files")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
LiveKit Voice Assistant Agent
A voice assistant using LiveKit Agents with OpenAI Realtime API.
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
    ChatContext
)
from livekit.plugins import openai, silero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MelbourneFitnessAgent(Agent):
    """Melbourne Fitness Studio voice assistant agent."""
    
    def __init__(self):
        # System prompt for the assistant
        system_prompt = """[Identity]  
        You are a friendly and articulate AI phone agent working for Melbourne Fitness Studio. You are calling a real customer who has already booked a tour of the studio. Your role is to simulate how a real front desk receptionist would handle a welcome and info-gathering call before their scheduled tour. You are not an AI demo assistant. This is not a demo scenario — act like a human staff member confirming the booking and collecting information to help personalize the visit.

        [Initial Greeting]
        Start the conversation by saying: "Hi, this is Sarah from Melbourne Fitness Studio. How's it going?"
        
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent."""
    
    try:
        # Connect to room
        logger.info(f"Connecting to room: {ctx.room.name}")
        await ctx.connect()
        logger.info(f"Successfully connected to room: {ctx.room.name}")
        
        # Parse room metadata
        is_phone_call = ctx.room.name.startswith("call-")
        phone_number = None
        
        if ctx.room.metadata:
            try:
                import json
                metadata = json.loads(ctx.room.metadata)
                is_phone_call = metadata.get("type") == "phone_call"
                phone_number = metadata.get("from")
                logger.info(f"Phone call from: {phone_number}")
            except Exception as e:
                logger.error(f"Failed to parse metadata: {e}")
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(0.5)
        
        # Create session using Realtime API
        logger.info("Creating session with OpenAI Realtime API")
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                api_key=os.getenv("OPENAI_API_KEY"),
                voice="coral",  # Options: alloy, echo, fable, onyx, nova, shimmer
                model="gpt-4o-mini-realtime-preview",  # or "gpt-4o-realtime-preview"
                temperature=0.7,
                # Optional: Enable turn detection
                # turn_detection=openai.realtime.TurnDetection(
                #     type="server_vad",
                #     threshold=0.5,
                #     prefix_padding_ms=300,
                #     silence_duration_ms=200
                # )
            ),
            vad=silero.VAD.load(),  # Still use VAD for better turn detection
        )
        
        # Start the session
        logger.info("Starting agent session...")
        await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
        logger.info("Agent session started successfully")
        
        # Remove this - the Realtime API will handle the greeting automatically
        # await session.generate_reply(
        #     instructions="Greet the user as Sarah from Melbourne Fitness Studio and ask how they're doing."
        # )
        
    except Exception as e:
        logger.error(f"Failed in entrypoint: {e}", exc_info=True)
        ctx.shutdown(reason=f"Error in entrypoint: {str(e)}")
        raise


async def console_mode():
    """Console mode for testing without LiveKit connection."""
    print("Melbourne Fitness Studio - Console Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    # For console mode, we'll use the regular LLM since Realtime API requires audio
    llm = openai.LLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
        base_url="https://api.openai.com/v1"
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
    required_vars = ["OPENAI_API_KEY"]
    
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