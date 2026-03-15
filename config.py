import json
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Google Cloud credentials bootstrap for Railway (or any env where you can't mount a file).
# Set GOOGLE_CREDENTIALS_JSON env var with the raw JSON content of your service account key.
# If GOOGLE_APPLICATION_CREDENTIALS is already pointing to a file, this is skipped.
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.environ.get("GOOGLE_CREDENTIALS_JSON"):
    _creds = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_creds, _tmp)
    _tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

SUPPORTED_LANGUAGES = {
    "ta-IN": {
        "name": "Tamil",
        "voice": "ta-IN-Wavenet-A",
        "label": "தமிழ்",
    },
    "hi-IN": {
        "name": "Hindi",
        "voice": "hi-IN-Wavenet-A",
        "label": "हिन्दी",
    },
    "es-ES": {
        "name": "Spanish",
        "voice": "es-ES-Wavenet-B",
        "label": "Español",
    },
    "ceb-PH": {
        "name": "Cebuano",
        "voice": "fil-PH-Wavenet-A",  # No Cebuano TTS voice; fall back to Filipino
        "tts_language_code": "fil-PH",  # Override for TTS
        "label": "Cebuano",
    },
    "fil-PH": {
        "name": "Tagalog",
        "voice": "fil-PH-Wavenet-A",
        "label": "Tagalog",
    },
}

DEFAULT_LANGUAGE_CODES = list(SUPPORTED_LANGUAGES.keys())

MAX_AUDIO_DURATION_SECONDS = 120

# Conversation memory: number of recent message pairs to keep per user
CONVERSATION_HISTORY_LIMIT = int(os.environ.get("CONVERSATION_HISTORY_LIMIT", "10"))

CLAUDE_MODEL = "claude-sonnet-4-0"
CLAUDE_MAX_TOKENS = 2048
