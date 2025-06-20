#!/usr/bin/env python3
"""
Setup script for LiveKit Voice Assistant
Helps configure the environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("📝 Creating .env file...")
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Deepgram API Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Cartesia API Configuration
CARTESIA_API_KEY=your_cartesia_api_key_here

# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here

# Optional: Environment-specific settings
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("✅ .env file created")
    print("⚠️  Please edit .env file with your actual API keys")


def download_silero_models():
    """Download Silero VAD and turn detection models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    models = {
        "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
        "silero_turn_detection.onnx": "https://github.com/snakers4/silero-vad/raw/master/files/silero_turn_detection.onnx"
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✅ {model_name} already exists")
            continue
        
        print(f"📥 Downloading {model_name}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            print("⚠️  Models will be downloaded automatically when needed")


def check_api_keys():
    """Check if API keys are configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = [
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
    ]
    
    missing_keys = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or value.startswith("your_") or value == "wss://your-livekit-server.com":
            missing_keys.append(key)
    
    if missing_keys:
        print("⚠️  Missing or placeholder API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease edit the .env file with your actual API keys")
        return False
    
    print("✅ All API keys are configured")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up LiveKit Voice Assistant...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Download models
    download_silero_models()
    
    # Check API keys
    api_keys_configured = check_api_keys()
    
    print("\n" + "=" * 50)
    if api_keys_configured:
        print("🎉 Setup complete! You can now run the voice assistant:")
        print("   Console mode: python agent.py console")
        print("   Development mode: python agent.py dev")
    else:
        print("⚠️  Setup complete, but please configure your API keys in .env file")
        print("   Then you can run: python agent.py console")


if __name__ == "__main__":
    main() 