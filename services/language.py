from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE_CODES


def get_language_name(language_code: str) -> str:
    lang = SUPPORTED_LANGUAGES.get(language_code)
    if lang:
        return lang["name"]
    return "Unknown"


def is_supported(language_code: str) -> bool:
    return language_code in SUPPORTED_LANGUAGES


def get_alternative_languages(primary: str | None = None) -> list[str]:
    """Return language codes to try for auto-detection."""
    if primary:
        return [c for c in DEFAULT_LANGUAGE_CODES if c != primary]
    return DEFAULT_LANGUAGE_CODES
