from __future__ import annotations

import logging

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE_CODES, AUTO_DETECT_LANGUAGE_CODES

logger = logging.getLogger(__name__)


def transcribe(audio_bytes: bytes, language_hint: str | None = None) -> tuple[str, str]:
    """Transcribe audio bytes to text using Google Cloud Speech-to-Text.

    If language_hint is provided (user chose a language), it is used as the sole
    language — giving accurate single-language transcription.

    If no hint, auto-detects by running STT independently for EACH supported
    language and picking the one with the highest confidence. This avoids
    Google STT's bias toward the primary language in multi-language mode.

    Returns (transcribed_text, detected_language_code).
    """
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)

    if language_hint and language_hint in SUPPORTED_LANGUAGES:
        logger.info("STT: using user-selected language %s", language_hint)
        transcript, _confidence = _recognize_single(client, audio, language_hint)
        return transcript, language_hint

    # Auto-detect: run STT once per language, compare confidence scores
    logger.info("STT: auto-detecting language among %s", AUTO_DETECT_LANGUAGE_CODES)

    best_transcript = ""
    best_confidence = -1.0
    best_language = AUTO_DETECT_LANGUAGE_CODES[0]

    for lang_code in AUTO_DETECT_LANGUAGE_CODES:
        transcript, confidence = _recognize_single(client, audio, lang_code)
        logger.info(
            "STT [%s]: confidence=%.3f transcript=%s",
            lang_code, confidence, transcript[:80] if transcript else "(empty)",
        )

        if confidence > best_confidence:
            best_confidence = confidence
            best_transcript = transcript
            best_language = lang_code

    logger.info("STT: winner language=%s confidence=%.3f", best_language, best_confidence)
    return best_transcript, best_language


def _recognize_single(
    client: speech.SpeechClient,
    audio: speech.RecognitionAudio,
    language_code: str,
) -> tuple[str, float]:
    """Run STT for a single language. Returns (transcript, confidence)."""
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return "", 0.0

    alt = response.results[0].alternatives[0]
    return alt.transcript, alt.confidence


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


def _normalize_language_code(detected: str) -> str:
    """Normalize a detected language code to match our supported codes."""
    for code in DEFAULT_LANGUAGE_CODES:
        if detected.lower() == code.lower():
            return code
    return DEFAULT_LANGUAGE_CODES[0]
