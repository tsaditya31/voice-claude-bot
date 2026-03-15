"""Tests for services.speech — transcribe and synthesize."""

from unittest.mock import MagicMock, patch

import pytest


# --- helpers to build Google STT/TTS mock responses ---

def _make_stt_result(transcript: str, confidence: float, language_code: str):
    """Build a mock SpeechRecognitionResult."""
    alt = MagicMock()
    alt.transcript = transcript
    alt.confidence = confidence

    result = MagicMock()
    result.alternatives = [alt]
    result.language_code = language_code
    return result


def _make_stt_response(results):
    resp = MagicMock()
    resp.results = results
    return resp


# ── transcribe with explicit language hint ──────────────────────────

@patch("services.speech.speech.SpeechClient")
def test_transcribe_with_hint_returns_transcript(mock_client_cls):
    """When a language hint is given, use it as primary and return the transcript."""
    from services.speech import transcribe

    mock_client = mock_client_cls.return_value
    mock_client.recognize.return_value = _make_stt_response(
        [_make_stt_result("வணக்கம்", 0.95, "ta-IN")]
    )

    text, lang = transcribe(b"fake-audio", language_hint="ta-IN")

    assert text == "வணக்கம்"
    assert lang == "ta-IN"
    # Should NOT use alternative_language_codes when hint is provided
    call_config = mock_client.recognize.call_args[1]["config"]
    assert call_config.language_code == "ta-IN"


@patch("services.speech.speech.SpeechClient")
def test_transcribe_with_hint_no_results(mock_client_cls):
    """When STT returns no results with a hint, return empty string + hint language."""
    from services.speech import transcribe

    mock_client = mock_client_cls.return_value
    mock_client.recognize.return_value = _make_stt_response([])

    text, lang = transcribe(b"silence", language_hint="hi-IN")

    assert text == ""
    assert lang == "hi-IN"


# ── transcribe with auto-detection ──────────────────────────────────

@patch("services.speech.speech.SpeechClient")
def test_transcribe_autodetect_picks_highest_confidence(mock_client_cls):
    """Without a hint, auto-detect runs batches and picks the highest confidence."""
    from services.speech import transcribe

    mock_client = mock_client_cls.return_value

    # First batch (4 langs): returns Spanish with high confidence
    batch1_response = _make_stt_response(
        [_make_stt_result("Hola mundo", 0.92, "es-ES")]
    )
    # Second batch (1 lang): returns something with lower confidence
    batch2_response = _make_stt_response(
        [_make_stt_result("some text", 0.40, "fil-PH")]
    )
    mock_client.recognize.side_effect = [batch1_response, batch2_response]

    text, lang = transcribe(b"fake-audio", language_hint=None)

    assert text == "Hola mundo"
    assert lang == "es-ES"
    assert mock_client.recognize.call_count == 2


@patch("services.speech.speech.SpeechClient")
def test_transcribe_autodetect_no_results_any_batch(mock_client_cls):
    """When all batches return no results, return empty string + default language."""
    from services.speech import transcribe
    from config import DEFAULT_LANGUAGE_CODES

    mock_client = mock_client_cls.return_value
    mock_client.recognize.return_value = _make_stt_response([])

    text, lang = transcribe(b"silence", language_hint=None)

    assert text == ""
    assert lang == DEFAULT_LANGUAGE_CODES[0]


@patch("services.speech.speech.SpeechClient")
def test_transcribe_autodetect_skips_empty_batches(mock_client_cls):
    """If first batch returns nothing and second returns results, use second."""
    from services.speech import transcribe

    mock_client = mock_client_cls.return_value
    empty = _make_stt_response([])
    hindi_result = _make_stt_response(
        [_make_stt_result("नमस्ते", 0.88, "hi-IN")]
    )
    mock_client.recognize.side_effect = [empty, hindi_result]

    text, lang = transcribe(b"fake-audio", language_hint=None)

    assert text == "नमस्ते"
    assert lang == "hi-IN"


# ── transcribe with unsupported hint falls back to auto-detect ──────

@patch("services.speech.speech.SpeechClient")
def test_transcribe_invalid_hint_falls_back_to_autodetect(mock_client_cls):
    """An unsupported language hint triggers auto-detect path."""
    from services.speech import transcribe

    mock_client = mock_client_cls.return_value
    mock_client.recognize.return_value = _make_stt_response(
        [_make_stt_result("Kumusta", 0.85, "fil-PH")]
    )

    text, lang = transcribe(b"fake-audio", language_hint="ja-JP")

    assert text == "Kumusta"
    assert lang == "fil-PH"


# ── synthesize ──────────────────────────────────────────────────────

@patch("services.speech.tts.TextToSpeechClient")
def test_synthesize_returns_audio_bytes(mock_tts_cls):
    """Synthesize returns audio content from TTS response."""
    from services.speech import synthesize

    mock_tts = mock_tts_cls.return_value
    mock_tts.synthesize_speech.return_value = MagicMock(audio_content=b"ogg-audio-data")

    result = synthesize("வணக்கம்", "ta-IN")

    assert result == b"ogg-audio-data"
    call_kwargs = mock_tts.synthesize_speech.call_args[1]
    assert call_kwargs["voice"].language_code == "ta-IN"
    assert call_kwargs["voice"].name == "ta-IN-Wavenet-A"


@patch("services.speech.tts.TextToSpeechClient")
def test_synthesize_cebuano_uses_filipino_tts(mock_tts_cls):
    """Cebuano should use fil-PH TTS language code (no native Cebuano voice)."""
    from services.speech import synthesize

    mock_tts = mock_tts_cls.return_value
    mock_tts.synthesize_speech.return_value = MagicMock(audio_content=b"audio")

    synthesize("Kumusta", "ceb-PH")

    call_kwargs = mock_tts.synthesize_speech.call_args[1]
    assert call_kwargs["voice"].language_code == "fil-PH"


@patch("services.speech.tts.TextToSpeechClient")
def test_synthesize_unsupported_language_raises(mock_tts_cls):
    """Synthesize should raise ValueError for unsupported language."""
    from services.speech import synthesize

    with pytest.raises(ValueError, match="Unsupported language"):
        synthesize("text", "xx-XX")


# ── helper functions ────────────────────────────────────────────────

def test_batch_language_codes():
    from services.speech import _batch_language_codes

    codes = ["a", "b", "c", "d", "e"]
    batches = _batch_language_codes(codes, batch_size=4)
    assert batches == [["a", "b", "c", "d"], ["e"]]


def test_batch_language_codes_exact_fit():
    from services.speech import _batch_language_codes

    codes = ["a", "b", "c", "d"]
    batches = _batch_language_codes(codes, batch_size=4)
    assert batches == [["a", "b", "c", "d"]]


def test_normalize_language_code():
    from services.speech import _normalize_language_code

    assert _normalize_language_code("ta-in") == "ta-IN"
    assert _normalize_language_code("ES-ES") == "es-ES"
    assert _normalize_language_code("fil-ph") == "fil-PH"


def test_normalize_language_code_unknown_returns_default():
    from services.speech import _normalize_language_code
    from config import DEFAULT_LANGUAGE_CODES

    assert _normalize_language_code("xx-XX") == DEFAULT_LANGUAGE_CODES[0]
