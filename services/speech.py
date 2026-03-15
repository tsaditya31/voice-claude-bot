from __future__ import annotations

import logging

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE_CODES

logger = logging.getLogger(__name__)


def transcribe(audio_bytes: bytes, language_hint: str | None = None) -> tuple[str, str]:
    """Transcribe audio bytes to text using Google Cloud Speech-to-Text.

    If language_hint is provided (user chose a language), it is used as the primary
    language with no alternatives — giving accurate single-language transcription.

    If no hint, auto-detects among all supported languages by running multiple
    recognition passes (Google STT limits alternatives to 3, so we batch).

    Returns (transcribed_text, detected_language_code).
    """
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)

    if language_hint and language_hint in SUPPORTED_LANGUAGES:
        # User explicitly chose a language — use it directly
        logger.info("STT: using user-selected language %s", language_hint)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_hint,
            enable_automatic_punctuation=True,
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            logger.warning("STT: no results for language %s", language_hint)
            return "", language_hint
        transcript = response.results[0].alternatives[0].transcript
        confidence = response.results[0].alternatives[0].confidence
        logger.info("STT: language=%s confidence=%.3f transcript=%s", language_hint, confidence, transcript[:80])
        return transcript, language_hint

    # Auto-detect: run recognition passes to cover all languages.
    # Google STT allows 1 primary + up to 3 alternative_language_codes = 4 per call.
    # With 5 languages we need 2 passes to cover all of them.
    logger.info("STT: auto-detecting language among %s", DEFAULT_LANGUAGE_CODES)
    batches = _batch_language_codes(DEFAULT_LANGUAGE_CODES, batch_size=4)

    best_transcript = ""
    best_confidence = -1.0
    best_language = DEFAULT_LANGUAGE_CODES[0]

    for batch in batches:
        primary = batch[0]
        alternatives = batch[1:]

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=primary,
            alternative_language_codes=alternatives,
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio)
        if not response.results:
            continue

        result = response.results[0]
        alt = result.alternatives[0]
        confidence = alt.confidence

        detected_raw = getattr(result, "language_code", primary)
        logger.info(
            "STT batch [%s]: detected=%s confidence=%.3f transcript=%s",
            "+".join(batch), detected_raw, confidence, alt.transcript[:80],
        )

        if confidence > best_confidence:
            best_confidence = confidence
            best_transcript = alt.transcript
            best_language = _normalize_language_code(detected_raw)

    logger.info("STT: final detected language=%s confidence=%.3f", best_language, best_confidence)
    return best_transcript, best_language


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


def _batch_language_codes(codes: list[str], batch_size: int = 4) -> list[list[str]]:
    """Split language codes into batches for STT calls (max 4 per call)."""
    batches = []
    for i in range(0, len(codes), batch_size):
        batches.append(codes[i : i + batch_size])
    return batches


def _normalize_language_code(detected: str) -> str:
    """Normalize a detected language code to match our supported codes."""
    for code in DEFAULT_LANGUAGE_CODES:
        if detected.lower() == code.lower():
            return code
    return DEFAULT_LANGUAGE_CODES[0]
