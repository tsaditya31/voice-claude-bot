"""Tests for config module."""

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE_CODES, AUTO_DETECT_LANGUAGE_CODES, CONVERSATION_HISTORY_LIMIT


def test_all_languages_have_required_keys():
    required_keys = {"name", "voice", "label"}
    for code, cfg in SUPPORTED_LANGUAGES.items():
        assert required_keys.issubset(cfg.keys()), f"{code} missing keys: {required_keys - cfg.keys()}"


def test_default_language_codes_matches_supported():
    assert set(DEFAULT_LANGUAGE_CODES) == set(SUPPORTED_LANGUAGES.keys())


def test_supported_languages_count():
    assert len(SUPPORTED_LANGUAGES) == 5


def test_cebuano_has_tts_override():
    """Cebuano has no native TTS voice, so it must override tts_language_code."""
    ceb = SUPPORTED_LANGUAGES["ceb-PH"]
    assert "tts_language_code" in ceb
    assert ceb["tts_language_code"] == "fil-PH"


def test_conversation_history_limit_is_positive():
    assert CONVERSATION_HISTORY_LIMIT > 0


def test_all_expected_languages_present():
    expected = {"ta-IN", "hi-IN", "es-ES", "ceb-PH", "fil-PH"}
    assert expected == set(SUPPORTED_LANGUAGES.keys())


def test_auto_detect_excludes_cebuano():
    """Cebuano is excluded from auto-detect due to inflated STT confidence."""
    assert "ceb-PH" not in AUTO_DETECT_LANGUAGE_CODES
    assert "ceb-PH" in DEFAULT_LANGUAGE_CODES  # still available for manual selection
