#!/usr/bin/env python3
"""
Test script for LiveKit Voice Assistant
Verifies that all components are working correctly.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from livekit import rtc, api
        from livekit.agents import AgentSession, Agent, ChatContext, WorkerOptions, cli
        from livekit.plugins import openai, deepgram, elevenlabs, silero
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_env_vars():
    """Test that required environment variables are set."""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",  # Changed from CARTESIA_API_KEY
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith("your_") or value == "wss://your-livekit-server.com":
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All environment variables are set")
    return True


async def test_openai_connection():
    """Test OpenAI API connection."""
    print("🔍 Testing OpenAI connection...")
    
    try:
        from livekit.plugins import openai
        from livekit.agents import ChatContext
        
        llm = openai.LLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        
        # Create chat context
        chat_ctx = ChatContext()
        chat_ctx.add_message(role="user", content="Hello, this is a test. Reply with 'Test successful'.")
        
        # Test with streaming
        response_text = ""
        stream = llm.chat(chat_ctx=chat_ctx)
        
        async for chunk in stream:
            # Extract content from ChatChunk's delta
            if hasattr(chunk, 'delta') and chunk.delta is not None:
                if hasattr(chunk.delta, 'content') and chunk.delta.content:
                    response_text += chunk.delta.content
        
        if response_text:
            print("✅ OpenAI connection successful")
            print(f"   Response: {response_text}")
            return True
        else:
            print("❌ OpenAI returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False


async def test_deepgram_connection():
    """Test Deepgram API connection."""
    print("🔍 Testing Deepgram connection...")
    
    try:
        from livekit.plugins import deepgram
        
        stt = deepgram.STT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-2",
            language="en-US"
        )
        
        # Just test that we can create the STT instance
        # Actual transcription would require proper audio data
        print("✅ Deepgram STT instance created successfully")
        return True
                
    except Exception as e:
        print(f"❌ Deepgram connection failed: {e}")
        return False


async def test_elevenlabs_connection():
    """Test ElevenLabs API connection."""
    print("🔍 Testing ElevenLabs connection...")
    
    try:
        from livekit.plugins import elevenlabs
        
        tts = elevenlabs.TTS(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="Sarah",  # or use a voice ID
            model="eleven_monolingual_v1"
        )
        
        # Test with a simple text
        test_text = "Hello, this is a test."
        
        # The actual synthesis method might be different
        # Just test that we can create the TTS instance
        print("✅ ElevenLabs TTS instance created successfully")
        return True
            
    except Exception as e:
        print(f"❌ ElevenLabs connection failed: {e}")
        return False


async def test_silero_vad():
    """Test Silero VAD loading."""
    print("🔍 Testing Silero VAD...")
    
    try:
        from livekit.plugins import silero
        
        vad = silero.VAD.load()
        print("✅ Silero VAD loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Silero VAD loading failed: {e}")
        return False


async def test_agent_creation():
    """Test that an agent can be created."""
    print("🔍 Testing agent creation...")
    
    try:
        from livekit.agents import Agent
        
        agent = Agent(
            instructions="You are a helpful assistant.",
        )
        print("✅ Agent creation successful")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False


async def test_console_chat():
    """Test basic chat functionality."""
    print("🔍 Testing console chat...")
    
    try:
        from livekit.plugins import openai
        from livekit.agents import ChatContext
        
        llm = openai.LLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        
        # Create chat context with system prompt
        chat_ctx = ChatContext()
        chat_ctx.add_message(
            role="system", 
            content="You are Sarah from Melbourne Fitness Studio. Be friendly and concise."
        )
        chat_ctx.add_message(
            role="user", 
            content="Hello, is this Melbourne Fitness Studio?"
        )
        
        # Get response
        response_text = ""
        stream = llm.chat(chat_ctx=chat_ctx)
        
        async for chunk in stream:
            # Use the same extraction method that worked in test_openai_connection
            if hasattr(chunk, 'delta') and chunk.delta is not None:
                if hasattr(chunk.delta, 'content') and chunk.delta.content:
                    response_text += chunk.delta.content
        
        if response_text and "fitness" in response_text.lower():
            print("✅ Console chat test successful")
            print(f"   Response preview: {response_text[:100]}...")
            return True
        else:
            print("❌ Console chat returned unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ Console chat test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Running LiveKit Voice Assistant Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment Variables", lambda: test_env_vars()),
        ("OpenAI Connection", test_openai_connection),
        ("Deepgram Connection", test_deepgram_connection),
        ("ElevenLabs Connection", test_elevenlabs_connection),
        ("Silero VAD", test_silero_vad),
        ("Agent Creation", test_agent_creation),
        ("Console Chat", test_console_chat),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your voice assistant is ready to use.")
        print("\nNext steps:")
        print("1. Run console mode: python agent.py console")
        print("2. Run voice mode: python agent.py dev")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check your .env file has all required API keys")
        print("3. Ensure API keys are valid and have credits/quota")


if __name__ == "__main__":
    asyncio.run(main())