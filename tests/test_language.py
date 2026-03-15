"""Tests for services.language."""

from services.language import get_language_name, is_supported, get_alternative_languages
from config import DEFAULT_LANGUAGE_CODES


def test_get_language_name_known():
    assert get_language_name("ta-IN") == "Tamil"
    assert get_language_name("hi-IN") == "Hindi"
    assert get_language_name("es-ES") == "Spanish"
    assert get_language_name("ceb-PH") == "Cebuano"
    assert get_language_name("fil-PH") == "Tagalog"


def test_get_language_name_unknown():
    assert get_language_name("xx-XX") == "Unknown"


def test_is_supported():
    assert is_supported("ta-IN") is True
    assert is_supported("es-ES") is True
    assert is_supported("ja-JP") is False
    assert is_supported("") is False


def test_get_alternative_languages_with_primary():
    alts = get_alternative_languages("ta-IN")
    assert "ta-IN" not in alts
    assert len(alts) == len(DEFAULT_LANGUAGE_CODES) - 1


def test_get_alternative_languages_without_primary():
    alts = get_alternative_languages()
    assert alts == DEFAULT_LANGUAGE_CODES
