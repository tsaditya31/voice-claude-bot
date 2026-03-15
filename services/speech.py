from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE_CODES


def transcribe(audio_bytes: bytes, language_hint: str | None = None) -> tuple[str, str]:
    """Transcribe audio bytes to text using Google Cloud Speech-to-Text.

    Returns (transcribed_text, detected_language_code).
    """
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_bytes)

    if language_hint and language_hint in SUPPORTED_LANGUAGES:
        primary = language_hint
        alternatives = [c for c in DEFAULT_LANGUAGE_CODES if c != primary]
    else:
        primary = DEFAULT_LANGUAGE_CODES[0]
        alternatives = DEFAULT_LANGUAGE_CODES[1:]

    # Google STT limits alternative_language_codes to 3
    alternatives = alternatives[:3]

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=primary,
        alternative_language_codes=alternatives,
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return "", primary

    result = response.results[0]
    transcript = result.alternatives[0].transcript
    detected_language = getattr(result, "language_code", primary)

    # Normalize language code (e.g. "ta-in" -> "ta-IN")
    for code in DEFAULT_LANGUAGE_CODES:
        if detected_language.lower() == code.lower():
            detected_language = code
            break

    return transcript, detected_language


def synthesize(text: str, language_code: str) -> bytes:
    """Convert text to speech using Google Cloud Text-to-Speech.

    Returns OGG_OPUS audio bytes (native Telegram voice format).
    """
    client = tts.TextToSpeechClient()

    lang_config = SUPPORTED_LANGUAGES.get(language_code)
    if not lang_config:
        raise ValueError(f"Unsupported language: {language_code}")

    synthesis_input = tts.SynthesisInput(text=text)

    # Some languages need a different TTS language code (e.g. Cebuano -> Filipino)
    tts_lang = lang_config.get("tts_language_code", language_code)

    voice = tts.VoiceSelectionParams(
        language_code=tts_lang,
        name=lang_config["voice"],
    )

    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.OGG_OPUS,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    return response.audio_content
