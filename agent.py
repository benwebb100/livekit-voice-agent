#!/usr/bin/env python3
"""
LiveKit Voice Assistant Agent
A multilingual voice assistant using LiveKit Agents with Deepgram STT, OpenAI GPT-4o-mini, and Cartesia TTS.
"""

import asyncio
import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from livekit import rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    Worker,
    Agent,
    ChatContext,
    ChatMessage,
    ChatResponse,
    AudioContext,
    AudioResponse,
    AudioChunk,
    AudioFormat,
    AudioEncoding,
    AudioSource,
    AudioSink,
    VADContext,
    VADResponse,
    TurnDetectionContext,
    TurnDetectionResponse,
)
from livekit.agents.llm import OpenAILLM
from livekit.agents.stt import DeepgramSTT
from livekit.agents.tts import ElevenLabsTTS
from livekit.agents.vad import SileroVAD
from livekit.agents.turn_detection import MultilingualTurnDetection
from livekit.agents.plugins import NoiseCancellationPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAssistantAgent(Agent):
    """Multilingual voice assistant agent with conversation capabilities."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.llm = OpenAILLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-realtime",
            temperature=0.7,
            max_tokens=1000
        )
        
        self.stt = DeepgramSTT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
            diarize=True
        )
        
        self.tts = ElevenLabsTTS(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice="aGkVQvWUZi16EH8aZJvT",
            model="eleven_monolingual_v1",
            stability=0.5,
            similarity_boost=0.75
        )
        
        self.vad = SileroVAD(
            model_path="silero_vad.onnx",
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=100,
            window_size_ms=30,
            step_size_ms=10
        )
        
        self.turn_detection = MultilingualTurnDetection(
            model_path="silero_turn_detection.onnx",
            threshold=0.5
        )
        
        # Initialize conversation context
        self.conversation_history = []
        self.user_name = None
        self.language = "en"
        
        # Greeting messages in multiple languages
        self.greetings = {
            "en": "Hello! I'm your AI voice assistant. How can I help you today?",
            "es": "¡Hola! Soy tu asistente de voz con IA. ¿Cómo puedo ayudarte hoy?",
            "fr": "Bonjour! Je suis votre assistant vocal IA. Comment puis-je vous aider aujourd'hui?",
            "de": "Hallo! Ich bin Ihr KI-Sprachassistent. Wie kann ich Ihnen heute helfen?",
            "it": "Ciao! Sono il tuo assistente vocale AI. Come posso aiutarti oggi?",
            "pt": "Olá! Sou seu assistente de voz com IA. Como posso ajudá-lo hoje?",
            "ja": "こんにちは！私はあなたのAI音声アシスタントです。今日はどのようにお手伝いできますか？",
            "ko": "안녕하세요! 저는 당신의 AI 음성 비서입니다. 오늘 어떻게 도와드릴까요?",
            "zh": "你好！我是你的AI语音助手。今天我能为您做些什么？",
            "ru": "Привет! Я ваш голосовой помощник с ИИ. Как я могу помочь вам сегодня?"
        }
        
        # System prompt for the assistant
        self.system_prompt = """[Identity]  
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
- End the call once all core information is collected and the tour is reconfirmed.

[Task & Conversation Flow]  
1. Greet the user:  
   - "Hi {{first_name}}, this is Sarah from Melbourne Fitness Studio — how's it going?"

2. Confirm the tour:  
   - You will be given `{{tour_date}}` in the format DD/MM/YYYY.  
   - When speaking, do **not say the year**.  
   - Convert it into a friendly format, such as:  
     > "I saw you've booked a tour with us for **next Tuesday**"  
     > or  
     > "...for **the 25th of July** — just wanted to give you a quick courtesy call to welcome you and confirm everything's locked in."  
   - Then ask:  
     > "Are you excited to get into some exercise?"

3. Rapport and expectations:  
   - "Are you mainly looking to use the gym, take our classes, or a bit of both?"

4. Fitness goals and motivations (ask one at a time):  
   - "Have you popped into the studio before?"  
     > If yes: "How was it?"  
     > If no: "Awesome — you've got a lot to look forward to!"  
   - "What kind of training or exercise do you enjoy?"  
   - "What kind of results are you hoping for?" (e.g., Trimming, Toning, Strength, Rehab)  
   - "When would you like to start seeing results? Do you have a month in mind?"  
   - "And when would you ideally like to get started working toward that goal?"

5. Schedule fit (each question separately):  
   - "How many times a week do you think you'd train?"  
   - "What time of day works best for you to train?"

6. Close and Reinforce Visit:

Awesome — sounds like a great fit!

Let's lock in a time for you to pop down to the club. When you come in, we'll have a quick chat about your goals and I'll show you around.

> "What day and time would suit you best to come in?"

→ Use `check_availability` to fetch open times based on the user's preferred window. Suggest the first 2–3 available options in a friendly tone.

If the user picks a time:  
→ Run `check_availability_cal` to confirm it's still open.  
If available, run `book_appointment_cal` and say:  
> "Perfect — I've locked that in for you. We'll see you at [Day, Time]!"

If unavailable:  
> "Ah, looks like that one's just been taken. Let me check the next best time for you…"

Finally, say:  
> "You know where we're located, right?"  
(If they say no: give a quick explanation or mention a nearby landmark.)

Wrap with:  
> "Looking forward to meeting you — we'll get everything lined up to help you hit your goals. See you soon!"

[Error Handling]  
- If user is unsure about their booking: "No worries, I can flag that for the team to confirm or reschedule with you."  
- If user is rushed: "Totally understand — I'll just note your preferences and we'll see you soon."  
- If user seems confused: "This is just a quick welcome and info call before your tour — nothing formal."

[End Condition]  
- After all core info is gathered and booking is confirmed, politely end the call.

[BusinessProfile]  
Business name: Melbourne Fitness Studio  
Business type: Fitness studio  
Location: 25 Carlisle Street, St Kilda, Melbourne VIC  
Top services: Strength & conditioning, group classes, personal training  
General tone/style: Friendly, supportive, results-focused

[FAQ]  
1. Do you offer group fitness classes?  
   → Yes! We run daily classes including HIIT, strength, and mobility options.  
2. Is there parking available?  
   → Yep — street parking is available nearby, and there's a small car park next door.  
3. Can I try a session before signing up?  
   → Yes, your first visit after the tour can be a free trial if you're keen.  
4. What types of training do you specialize in?  
   → We focus on weight loss, mobility, and strength for everyday people — no ego, just progress.  
5. Do you have personal trainers?  
   → Absolutely. We have multiple qualified PTs that can support your goals 1-on-1."""
    
    async def on_chat(self, ctx: ChatContext) -> ChatResponse:
        """Handle chat messages from the user."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": ctx.message.content
            })
            
            # Prepare messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-10:])  # Keep last 10 messages for context
            
            # Get response from LLM
            response = await self.llm.chat(messages)
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            return ChatResponse(content=response.content)
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return ChatResponse(content="I apologize, but I encountered an error. Please try again.")
    
    async def on_audio(self, ctx: AudioContext) -> AudioResponse:
        """Handle audio input and generate audio response."""
        try:
            # Process audio through STT
            transcription = await self.stt.transcribe(ctx.audio)
            
            if not transcription.text.strip():
                return AudioResponse()
            
            # Detect language and update if needed
            detected_lang = transcription.language or "en"
            if detected_lang != self.language:
                self.language = detected_lang
                logger.info(f"Language detected: {detected_lang}")
            
            # Create chat context for the transcribed text
            chat_ctx = ChatContext(
                message=ChatMessage(
                    content=transcription.text,
                    role="user"
                )
            )
            
            # Get text response
            chat_response = await self.on_chat(chat_ctx)
            
            if not chat_response.content:
                return AudioResponse()
            
            # Convert text response to speech
            audio_data = await self.tts.synthesize(chat_response.content)
            
            return AudioResponse(
                audio=AudioChunk(
                    data=audio_data,
                    format=AudioFormat(
                        encoding=AudioEncoding.LINEAR16,
                        sample_rate=24000,
                        channels=1
                    )
                )
            )
            
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return AudioResponse()
    
    async def on_vad(self, ctx: VADContext) -> VADResponse:
        """Handle Voice Activity Detection."""
        try:
            result = await self.vad.detect(ctx.audio)
            return VADResponse(
                is_speech=result.is_speech,
                confidence=result.confidence
            )
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return VADResponse(is_speech=False, confidence=0.0)
    
    async def on_turn_detection(self, ctx: TurnDetectionContext) -> TurnDetectionResponse:
        """Handle multilingual turn detection."""
        try:
            result = await self.turn_detection.detect(ctx.audio)
            return TurnDetectionResponse(
                is_turn=result.is_turn,
                confidence=result.confidence,
                language=result.language
            )
        except Exception as e:
            logger.error(f"Error in turn detection: {e}")
            return TurnDetectionResponse(is_turn=False, confidence=0.0)
    
    async def on_start(self):
        """Called when the agent starts."""
        logger.info("Voice Assistant Agent started")
        
        # Send initial greeting
        greeting = self.greetings.get(self.language, self.greetings["en"])
        logger.info(f"Greeting user: {greeting}")
        
        # Synthesize and play greeting
        try:
            audio_data = await self.tts.synthesize(greeting)
            # Note: In a real implementation, you'd want to play this audio
            # For now, we'll just log that we would play it
            logger.info("Greeting audio synthesized successfully")
        except Exception as e:
            logger.error(f"Error synthesizing greeting: {e}")


async def main():
    """Main entry point for the voice assistant."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py [console|dev]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY", 
        "ELEVENLABS_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        sys.exit(1)
    
    # Create agent instance
    agent = VoiceAssistantAgent()
    
    if mode == "console":
        # Console mode for testing
        print("Voice Assistant Console Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        # Send initial greeting
        greeting = agent.greetings["en"]
        print(f"Assistant: {greeting}")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Assistant: Goodbye! Have a great day!")
                    break
                
                if not user_input:
                    continue
                
                # Process through chat
                chat_ctx = ChatContext(
                    message=ChatMessage(content=user_input, role="user")
                )
                response = await agent.on_chat(chat_ctx)
                print(f"Assistant: {response.content}")
                
            except KeyboardInterrupt:
                print("\nAssistant: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif mode == "dev":
        # Development mode with LiveKit
        print("Starting Voice Assistant in development mode...")
        
        # Configure worker options
        options = WorkerOptions(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            agent=agent,
            plugins=[NoiseCancellationPlugin()]  # Enable noise cancellation
        )
        
        # Create and start worker
        worker = Worker(options)
        
        try:
            await worker.run()
        except KeyboardInterrupt:
            logger.info("Shutting down worker...")
        finally:
            await worker.shutdown()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: console, dev")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 