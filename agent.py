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

import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.plugins import (
    openai,
    silero,
)

# Load .env.local
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-realtime-fitness")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class FitnessOutboundAgent(Agent):
    def __init__(self, *, name: str, tour_time: str, dial_info: dict[str, Any]):
        super().__init__(instructions=f"""
        You are a friendly front desk receptionist from Melbourne Fitness Studio calling to confirm a tour appointment with a customer named {name} scheduled for {tour_time}.

        [Your Personality]
        - You sound like Sarah from Melbourne Fitness.
        - Be cheerful, energetic, and supportive.
        - Always speak like a real human — avoid robotic or overly formal tone.
        - Ask one question at a time. Wait for the user to reply before continuing.
        - Don’t rush. Build rapport and gather details for the visit.

        [When the customer requests a human agent]
        - Confirm with them first, then use the transfer_call tool.

        [If you reach voicemail]
        - Wait until the voicemail greeting ends, then use the detected_answering_machine tool.

        [DO NOT]
        - Never say you’re an AI.
        - Never mention this is a simulation or test.
        - Never offer real-time booking or integrations.

        End politely once all key info is confirmed.
        """)
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        transfer_to = self.dial_info.get("transfer_to")
        if not transfer_to:
            return "Transfer number not available."

        logger.info(f"Transferring call to: {transfer_to}")
        await ctx.session.generate_reply(instructions="Sure, let me transfer you now.")
        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )
        except Exception as e:
            logger.error(f"Error during transfer: {e}")
            await ctx.session.generate_reply(instructions="Sorry, there was an issue transferring you.")
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        logger.info(f"Call ended by user: {self.participant.identity}")
        if ctx.session.current_speech:
            await ctx.session.current_speech.wait_for_playout()
        await self.hangup()

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        logger.info(f"Voicemail detected for: {self.participant.identity}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    dial_info = json.loads(ctx.job.metadata or '{}')
    participant_identity = phone_number = dial_info.get("phone_number")

    agent = FitnessOutboundAgent(
        name="Jayden",
        tour_time="Tuesday at 5 PM",
        dial_info=dial_info,
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=None,  # OpenAI Realtime handles STT internally
        llm=openai.realtime.RealtimeModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-realtime-preview",
            voice="nova",  # or "coral", "onyx", etc.
            temperature=0.7,
        ),
    )

    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
            ),
        )
    )

    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
            )
        )

        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"Participant joined: {participant.identity}")
        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"Failed to dial: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller-fitness",
        )
    )
