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
        from livekit import rtc
        from livekit.agents import Agent, ChatContext, ChatMessage, ChatResponse
        from livekit.agents.llm import OpenAILLM
        from livekit.agents.stt import DeepgramSTT
        from livekit.agents.tts import CartesiaTTS
        from livekit.agents.vad import SileroVAD
        from livekit.agents.turn_detection import MultilingualTurnDetection
        from livekit.agents.plugins import NoiseCancellationPlugin
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
        "CARTESIA_API_KEY",
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
        from livekit.agents.llm import OpenAILLM
        
        llm = OpenAILLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100
        )
        
        # Test with a simple message
        messages = [{"role": "user", "content": "Hello, this is a test."}]
        response = await llm.chat(messages)
        
        if response.content:
            print("✅ OpenAI connection successful")
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
        from livekit.agents.stt import DeepgramSTT
        
        stt = DeepgramSTT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-2",
            language="en-US"
        )
        
        # Create a simple test audio (silence)
        import numpy as np
        test_audio = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second of silence
        
        # Note: This might fail with silence, but we're testing the connection
        try:
            result = await stt.transcribe(test_audio)
            print("✅ Deepgram connection successful")
            return True
        except Exception as e:
            if "no speech detected" in str(e).lower():
                print("✅ Deepgram connection successful (no speech in test audio)")
                return True
            else:
                print(f"❌ Deepgram transcription failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Deepgram connection failed: {e}")
        return False


async def test_cartesia_connection():
    """Test Cartesia API connection."""
    print("🔍 Testing Cartesia connection...")
    
    try:
        from livekit.agents.tts import CartesiaTTS
        
        tts = CartesiaTTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice="nova",
            model="cartesia-1"
        )
        
        # Test with a simple text
        test_text = "Hello, this is a test."
        audio_data = await tts.synthesize(test_text)
        
        if audio_data and len(audio_data) > 0:
            print("✅ Cartesia connection successful")
            return True
        else:
            print("❌ Cartesia returned empty audio")
            return False
            
    except Exception as e:
        print(f"❌ Cartesia connection failed: {e}")
        return False


async def test_agent_creation():
    """Test that the voice assistant agent can be created."""
    print("🔍 Testing agent creation...")
    
    try:
        from agent import VoiceAssistantAgent
        
        agent = VoiceAssistantAgent()
        print("✅ Agent creation successful")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False


async def test_console_mode():
    """Test console mode functionality."""
    print("🔍 Testing console mode...")
    
    try:
        from agent import VoiceAssistantAgent
        from livekit.agents import ChatContext, ChatMessage
        
        agent = VoiceAssistantAgent()
        
        # Test a simple chat interaction
        chat_ctx = ChatContext(
            message=ChatMessage(content="Hello, this is a test.", role="user")
        )
        
        response = await agent.on_chat(chat_ctx)
        
        if response.content:
            print("✅ Console mode test successful")
            return True
        else:
            print("❌ Console mode returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Console mode test failed: {e}")
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
        ("Cartesia Connection", test_cartesia_connection),
        ("Agent Creation", test_agent_creation),
        ("Console Mode", test_console_mode),
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
        print("🎉 All tests passed! Your voice assistant is ready to use.")
        print("   Run: python agent.py console")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   Make sure all API keys are configured correctly.")


if __name__ == "__main__":
    asyncio.run(main()) 