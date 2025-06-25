# #!/usr/bin/env python3
# """
# LiveKit Voice Assistant Agent
# A voice assistant using LiveKit Agents with OpenAI Realtime API.
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
#     AutoSubscribe,
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     ChatContext
# )
# from livekit.plugins import openai, silero

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()


# class MelbourneFitnessAgent(Agent):
#     """Melbourne Fitness Studio voice assistant agent."""
    
#     def __init__(self):
#         # System prompt for the assistant
#         system_prompt = """[Identity]  
#         You are a friendly and articulate AI phone agent working for Melbourne Fitness Studio. You are calling a real customer who has already booked a tour of the studio. Your role is to simulate how a real front desk receptionist would handle a welcome and info-gathering call before their scheduled tour. You are not an AI demo assistant. This is not a demo scenario — act like a human staff member confirming the booking and collecting information to help personalize the visit.

#         [Initial Greeting]
#         Start the conversation by saying: "Hi, this is Sarah from Melbourne Fitness Studio. How's it going?"
        
#         [Style]  
#         - Speak clearly and naturally.  
#         - Use a warm, conversational tone.  
#         - Avoid robotic or overly formal speech.  
#         - Use casual phrasing and human-like timing (e.g., "Alright…", "Cool!").  
#         - Keep sentences short and natural.  
#         - Reflect the tone of the business: supportive, friendly, energetic

#         [Response Guidelines]  
#         - Stay in character as a real staff member from Melbourne Fitness Studio.  
#         - Always ask one question at a time and wait for a response before moving to the next.  
#         - Do not rush the call — you are building rapport and excitement.  
#         - If the user needs to reschedule, acknowledge politely and suggest someone from the team will follow up.  
#         - Never say you're an AI.  
#         - Do not offer real-time bookings or calendar integration — this is a simulation only.  
#         - End the call once all core information is collected and the tour is reconfirmed."""
        
#         super().__init__(instructions=system_prompt)


# async def entrypoint(ctx: JobContext):
#     """Main entry point for the agent."""
    
#     try:
#         # Connect to room
#         logger.info(f"Connecting to room: {ctx.room.name}")
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"Successfully connected to room: {ctx.room.name}")
        
#         # Parse room metadata
#         # is_phone_call = ctx.room.name.startswith("call-")
#         # phone_number = None
#         phone_number = ctx.job.metadata
#         logger.info(f"dialing {phone_number} to room {ctx.room.name}")
        
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
        
#         # Create session using Realtime API
#         logger.info("Creating session with OpenAI Realtime API")
#         session = AgentSession(
#             llm=openai.realtime.RealtimeModel(
#                 api_key=os.getenv("OPENAI_API_KEY"),
#                 voice="coral",  # Options: alloy, echo, fable, onyx, nova, shimmer
#                 model="gpt-4o-mini-realtime-preview",  # or "gpt-4o-realtime-preview"
#                 temperature=0.7,
#             ),
#             vad=silero.VAD.load(),  # Still use VAD for better turn detection
#         )
        
#         # Start the session
#         logger.info("Starting agent session...")
#         await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
#         logger.info("Agent session started successfully")
        
#         # Remove this - the Realtime API will handle the greeting automatically
#         # await session.generate_reply(
#         #     instructions="Greet the user as Sarah from Melbourne Fitness Studio and ask how they're doing."
#         # )
        
#     except Exception as e:
#         logger.error(f"Failed in entrypoint: {e}", exc_info=True)
#         ctx.shutdown(reason=f"Error in entrypoint: {str(e)}")
#         raise


# async def console_mode():
#     """Console mode for testing without LiveKit connection."""
#     print("Melbourne Fitness Studio - Console Mode")
#     print("Type 'quit' to exit")
#     print("-" * 50)
    
#     # For console mode, we'll use the regular LLM since Realtime API requires audio
#     llm = openai.LLM(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-4o-mini",
#         temperature=0.7,
#         base_url="https://api.openai.com/v1"
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
#     required_vars = ["OPENAI_API_KEY"]
    
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
LiveKit Voice Assistant Agent for Melbourne Fitness Studio
A voice assistant using LiveKit Agents with OpenAI Realtime API and Twilio integration.
"""

import asyncio
import os
import sys
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from time import perf_counter

from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    ChatContext,
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

# Get SIP trunk ID from environment
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


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
        - End the call once all core information is collected and the tour is reconfirmed.

        Your interface with the user will be voice. You will be on a call with a customer who has an upcoming tour appointment. Your goal is to confirm the appointment details and gather information to personalize their visit."""
        
        super().__init__(instructions=system_prompt)


class MelbourneFitnessActions:
    """
    Handle call actions for Melbourne Fitness Studio
    """

    def __init__(
        self, *, api: api.LiveKitAPI, participant: rtc.RemoteParticipant, room: rtc.Room
    ):
        self.api = api
        self.participant = participant
        self.room = room

    async def hangup(self):
        """Helper method to hang up the call"""
        try:
            await self.api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=self.room.name,
                    identity=self.participant.identity,
                )
            )
        except Exception as e:
            # User may have already hung up, this error can be ignored
            logger.info(f"Received error while ending call: {e}")

    async def end_call(self):
        """Called when the conversation is complete or user wants to end the call"""
        logger.info(f"Ending the call for {self.participant.identity}")
        await self.hangup()

    async def confirm_tour_appointment(self, date: str, time: str, customer_name: str = None):
        """Called when confirming the customer's tour appointment details"""
        logger.info(
            f"Confirming tour appointment for {self.participant.identity}: {customer_name} on {date} at {time}"
        )
        return f"Tour appointment confirmed for {date} at {time}"

    async def schedule_callback(self, preferred_time: str, reason: str = None):
        """Called when the customer needs to reschedule or wants a callback"""
        logger.info(
            f"Scheduling callback for {self.participant.identity} at {preferred_time}. Reason: {reason}"
        )
        return f"Callback scheduled for {preferred_time}"

    async def detected_answering_machine(self):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"Detected answering machine for {self.participant.identity}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent."""
    
    try:
        logger.info(f"Connecting to room: {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        
        user_identity = "phone_user"
        # The phone number to dial is provided in the job metadata
        phone_number = ctx.job.metadata
        logger.info(f"Dialing {phone_number} to room {ctx.room.name}")
        
        # Create SIP participant to start dialing
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=user_identity,
            )
        )
        
        # Wait for participant to join
        participant = await ctx.wait_for_participant(identity=user_identity)
        
        # Monitor call status and start session when call is active
        start_time = perf_counter()
        call_connected = False
        
        while perf_counter() - start_time < 30:  # 30 second timeout
            call_status = participant.attributes.get("sip.callStatus")
            
            if call_status == "active" and not call_connected:
                logger.info("User has picked up the phone, starting session")
                call_connected = True
                
                # Start the Realtime session
                await start_realtime_session(ctx, participant)
                return
                
            elif call_status == "automation":
                # DTMF dialing state (extensions, PIN entry, etc.)
                logger.info("Call in automation state (DTMF)")
                pass
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
                logger.info("User rejected the call, exiting job")
                break
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
                logger.info("User did not pick up, exiting job")
                break
                
            await asyncio.sleep(0.1)
        
        if not call_connected:
            logger.info("Session timed out or call failed, exiting job")
            ctx.shutdown()
        
    except Exception as e:
        logger.error(f"Failed in entrypoint: {e}", exc_info=True)
        ctx.shutdown(reason=f"Error in entrypoint: {str(e)}")
        raise


async def start_realtime_session(ctx: JobContext, participant: rtc.RemoteParticipant):
    """Start the Realtime API session for the connected call."""
    
    try:
        # Wait a moment for connection to stabilize
        await asyncio.sleep(0.5)
        
        # Create function context for call actions
        actions = MelbourneFitnessActions(
            api=ctx.api,
            participant=participant,
            room=ctx.room
        )
        
        # Create session using Realtime API
        logger.info("Creating session with OpenAI Realtime API")
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                api_key=os.getenv("OPENAI_API_KEY"),
                voice="coral",  # Options: alloy, echo, fable, onyx, nova, shimmer, coral
                model="gpt-4o-mini-realtime-preview",  # or "gpt-4o-realtime-preview"
                temperature=0.7,
            ),
            vad=silero.VAD.load(),  # Use VAD for better turn detection
        )
        
        # Start the session
        logger.info("Starting agent session...")
        await session.start(room=ctx.room, agent=MelbourneFitnessAgent())
        logger.info("Agent session started successfully")
        
        # The Realtime API will handle the greeting automatically based on the system prompt
        
    except Exception as e:
        logger.error(f"Failed to start Realtime session: {e}", exc_info=True)
        ctx.shutdown(reason=f"Error in Realtime session: {str(e)}")
        raise


def prewarm(proc: JobProcess):
    """Prewarm function to load models before handling calls."""
    # Load VAD model for better performance
    proc.userdata["vad"] = silero.VAD.load()


async def console_mode():
    """Console mode for testing without LiveKit connection."""
    print("Melbourne Fitness Studio - Console Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    # For console mode, use regular LLM since Realtime API requires audio
    llm_instance = openai.LLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # Create agent for system prompt
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
            stream = llm_instance.chat(chat_ctx=chat_ctx)
            
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
            "LIVEKIT_API_SECRET",
            "SIP_OUTBOUND_TRUNK_ID"
        ])
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        sys.exit(1)
    
    # Validate SIP trunk ID format
    if mode in ["dev", "start"]:
        if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
            raise ValueError(
                "SIP_OUTBOUND_TRUNK_ID is not set or invalid format (should start with 'ST_')"
            )
    
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
            # Give this agent a name for API dispatch
            agent_name="melbourne-fitness-outbound",
            # Prewarm by loading the VAD model
            prewarm_fnc=prewarm,
        )
        
        # Pass the mode to CLI
        cli.run_app(worker_options)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: console, dev, start, download-files")
        sys.exit(1)


if __name__ == "__main__":
    main()