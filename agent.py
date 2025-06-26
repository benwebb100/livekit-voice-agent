from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any
from datetime import datetime, timedelta

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class MelbourneFitnessAgent(Agent):
    def __init__(
        self,
        *,
        first_name: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            You are Sarah, a friendly team member at Melbourne Fitness Studio calling a lead who previously enquired online about joining the club but never booked a tour. Your job is to reconnect, build quick rapport, offer them a free 1-week "mini membership" trial, and book them in for a visit to get started.

            You are NOT an AI demo assistant. This is NOT a test call — act like a real team member checking in.

            BUSINESS PROFILE:
            - Business name: Melbourne Fitness Studio
            - Location: 25 Carlisle Street, St Kilda, Melbourne VIC
            - Services: Strength & conditioning, group classes, personal training
            - Tone: Friendly, supportive, results-focused

            STYLE GUIDELINES:
            - Speak clearly and naturally with a warm, upbeat, conversational tone
            - Avoid robotic or overly formal speech
            - Ask one question at a time and pause to listen
            - Keep responses brief and engaging
            - Stay in character as Sarah from Melbourne Fitness Studio

            CONVERSATION FLOW:
            1. Greet: "Hi {first_name}, it's Sarah from Melbourne Fitness Studio — how's your day going?"
            2. Re-engagement: "You popped your details in with us a little while ago — I just wanted to quickly check in. Were you still looking to start something for your health and fitness?"
            3. Discovery: Ask about their goals and current training situation
            4. Offer mini membership: Mention the 1-week trial as a low-commitment way to try things out
            5. Schedule: Get them booked in for a visit this week
            6. Wrap up: Confirm location and what to expect

            ERROR HANDLING:
            - If confused about enquiry: "Totally fine — maybe someone used your number or you popped your name in somewhere online. I'll keep it super quick."
            - If rushed/unsure: "No worries at all — I'll let the team know, and we can follow up another time if you're interested."

            Keep the call brief, friendly, and focused on booking a visit. Use the available tools to check availability and book appointments.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.first_name = first_name
        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human team member if needed"""
        transfer_to = self.dial_info.get("transfer_to")
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        await ctx.session.generate_reply(
            instructions="let the user know you'll be transferring them to another team member"
        )

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )
            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call to another team member."
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the conversation naturally ends"""
        logger.info(f"ending the call for {self.participant.identity}")

        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await self.hangup()

    @function_tool()
    async def check_availability(
        self,
        ctx: RunContext,
        preferred_day: str = "",
        preferred_time: str = "",
    ):
        """Check available appointment times for gym visits and tours
        
        Args:
            preferred_day: The day they prefer (e.g., "Monday", "this week", "tomorrow")
            preferred_time: The time they prefer (e.g., "morning", "afternoon", "evening")
        """
        logger.info(f"checking availability for {self.participant.identity} - day: {preferred_day}, time: {preferred_time}")
        
        # Simulate checking availability
        await asyncio.sleep(2)
        
        # Return mock available times based on preferences
        if "morning" in preferred_time.lower():
            available_times = ["9:00 AM", "10:30 AM", "11:00 AM"]
        elif "afternoon" in preferred_time.lower():
            available_times = ["2:00 PM", "3:30 PM", "4:00 PM"]
        elif "evening" in preferred_time.lower():
            available_times = ["6:00 PM", "7:00 PM", "7:30 PM"]
        else:
            available_times = ["10:30 AM", "2:00 PM", "6:00 PM"]
        
        return {
            "available_times": available_times,
            "day": preferred_day or "this week"
        }

    @function_tool()
    async def book_appointment(
        self,
        ctx: RunContext,
        day: str,
        time: str,
        appointment_type: str = "gym tour and trial signup",
    ):
        """Book an appointment for the prospect to visit the gym
        
        Args:
            day: The day of the appointment (e.g., "Monday", "Tuesday")
            time: The time of the appointment (e.g., "10:30 AM", "2:00 PM")
            appointment_type: Type of appointment (default: gym tour and trial signup)
        """
        logger.info(f"booking appointment for {self.participant.identity} on {day} at {time}")
        
        # Simulate booking process
        await asyncio.sleep(2)
        
        return {
            "status": "confirmed",
            "day": day,
            "time": time,
            "location": "25 Carlisle Street, St Kilda, Melbourne VIC",
            "appointment_type": appointment_type,
            "message": "Perfect — I've locked that in for you. We'll see you then!"
        }

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"detected answering machine for {self.participant.identity}")
        
        # Leave a brief, friendly voicemail
        await ctx.session.generate_reply(
            instructions="""Leave a brief, friendly voicemail message as Sarah from Melbourne Fitness Studio. 
            Say something like: 'Hi, it's Sarah from Melbourne Fitness Studio. I was just checking in about your enquiry. 
            Give us a call back when you get a chance on [studio number]. Have a great day!'"""
        )
        
        # Wait for message to finish playing
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
            
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - first_name: the prospect's first name
    # - transfer_to: optional phone number to transfer to if needed
    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]
    first_name = dial_info.get("first_name", "there")

    # Create the Melbourne Fitness Studio agent
    agent = MelbourneFitnessAgent(
        first_name=first_name,
        dial_info=dial_info,
    )

    # Create agent session with voice capabilities
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.realtime.RealtimeModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-realtime-preview",
            voice="coral",  # or "coral", "onyx", etc.
            temperature=0.7,
        ),
    )

    # Start the session first before dialing
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # Enable noise cancellation for clear phone calls
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    # Start dialing the prospect
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # Wait until the call is answered
                wait_until_answered=True,
            )
        )

        # Wait for the session to start and participant to join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="melbourne-fitness-caller",
        )
    )