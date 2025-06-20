# LiveKit Voice Assistant

A multilingual voice assistant built with LiveKit Agents, featuring Deepgram speech-to-text, OpenAI GPT-4o-mini for natural language processing, and Cartesia text-to-speech. The assistant includes Silero VAD (Voice Activity Detection) and multilingual turn detection capabilities.

## Features

- **Multilingual Support**: Communicates in 10+ languages with automatic language detection
- **High-Quality Speech Recognition**: Uses Deepgram's Nova-2 model for accurate transcription
- **Natural Language Processing**: Powered by OpenAI's GPT-4o-mini for intelligent responses
- **Realistic Text-to-Speech**: Cartesia TTS for natural-sounding voice output
- **Voice Activity Detection**: Silero VAD for efficient speech detection
- **Turn Detection**: Multilingual turn detection for natural conversation flow
- **Noise Cancellation**: LiveKit Cloud noise cancellation plugin
- **Conversation Memory**: Maintains context across conversation turns

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - OpenAI
  - Deepgram
  - Cartesia
  - LiveKit

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd voice-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your actual API keys:
```env
# OpenAI API Configuration
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
```

## Usage

### Console Mode (Testing)

Run the assistant in console mode for testing without audio:

```bash
python agent.py console
```

This mode allows you to interact with the assistant via text input/output.

### Development Mode (Full Voice)

Run the assistant with full voice capabilities:

```bash
python agent.py dev
```

This mode connects to LiveKit and enables full voice interaction with:
- Speech-to-text transcription
- Natural language processing
- Text-to-speech synthesis
- Voice activity detection
- Turn detection
- Noise cancellation

## Supported Languages

The assistant supports the following languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Russian (ru)

## Configuration

### Voice Assistant Settings

You can customize the assistant behavior by modifying the `VoiceAssistantAgent` class:

- **LLM Settings**: Model, temperature, max tokens
- **STT Settings**: Model, language, formatting options
- **TTS Settings**: Voice, model, speed
- **VAD Settings**: Threshold, duration parameters
- **Turn Detection**: Model, threshold

### Audio Format

The assistant uses the following audio format:
- Encoding: LINEAR16
- Sample Rate: 24000 Hz
- Channels: 1 (Mono)

## API Keys Setup

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get your API key
3. Add the key to your `.env` file

### Deepgram
1. Visit [Deepgram](https://deepgram.com/)
2. Sign up and get your API key
3. Add the key to your `.env` file

### Cartesia
1. Visit [Cartesia](https://cartesia.ai/)
2. Create an account and get your API key
3. Add the key to your `.env` file

### LiveKit
1. Visit [LiveKit Cloud](https://livekit.io/)
2. Create a project and get your credentials
3. Add the URL, API key, and secret to your `.env` file

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in your `.env` file
2. **Audio Issues**: Check your microphone permissions and audio settings
3. **Network Issues**: Verify your internet connection and LiveKit server accessibility
4. **Model Downloads**: Silero models will be downloaded automatically on first use

### Logging

The assistant uses Python's logging module. You can adjust the log level in your `.env` file:
- DEBUG: Detailed debugging information
- INFO: General information (default)
- WARNING: Warning messages
- ERROR: Error messages only

## Development

### Project Structure

```
voice-assistant/
├── agent.py              # Main voice assistant implementation
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .env                 # Environment variables (create from .env.example)
```

### Extending the Assistant

You can extend the assistant by:
1. Adding new language support in the `greetings` dictionary
2. Modifying the system prompt for different personalities
3. Adding new conversation handlers
4. Implementing custom audio processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the LiveKit Agents documentation
3. Open an issue on the repository 