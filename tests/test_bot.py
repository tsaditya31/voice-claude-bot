"""Tests for bot.py handlers using mocked Telegram and services."""

from unittest.mock import AsyncMock, MagicMock, patch
import io

import pytest


# ── helpers ─────────────────────────────────────────────────────────

def _make_update(user_id=1, voice_duration=5, has_voice=True):
    """Build a mock Telegram Update with a voice message."""
    update = AsyncMock()
    update.effective_user.id = user_id
    update.message.reply_text = AsyncMock()
    update.message.reply_voice = AsyncMock()

    if has_voice:
        voice = MagicMock()
        voice.duration = voice_duration
        voice.get_file = AsyncMock()
        voice.get_file.return_value.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake-ogg"))
        update.message.voice = voice
    else:
        update.message.voice = None

    return update


def _make_context(user_data=None):
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    return ctx


# ── handle_voice ────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("bot._ogg_to_wav", return_value=b"fake-wav")
@patch("bot.speech")
@patch("bot.claude")
async def test_handle_voice_autodetect_flow(mock_claude, mock_speech, mock_ogg):
    """Full flow: no language set → auto-detect → respond."""
    from bot import handle_voice

    mock_speech.transcribe.return_value = ("Hola", "es-ES")
    mock_claude.ask.return_value = ("Respuesta completa", "Resumen corto")
    mock_speech.synthesize.return_value = b"ogg-audio"

    update = _make_update(user_id=1)
    context = _make_context({})  # no language set → auto-detect

    await handle_voice(update, context)

    # Should call transcribe with None hint (auto-detect)
    mock_speech.transcribe.assert_called_once_with(b"fake-wav", None)
    mock_claude.ask.assert_called_once_with("Hola", "es-ES", 1)
    mock_speech.synthesize.assert_called_once_with("Respuesta completa", "es-ES")
    update.message.reply_voice.assert_called_once()


@pytest.mark.asyncio
@patch("bot._ogg_to_wav", return_value=b"fake-wav")
@patch("bot.speech")
@patch("bot.claude")
async def test_handle_voice_with_language_hint(mock_claude, mock_speech, mock_ogg):
    """When user has set a language, pass it as hint."""
    from bot import handle_voice

    mock_speech.transcribe.return_value = ("நன்றி", "ta-IN")
    mock_claude.ask.return_value = ("பதில்", "சுருக்கம்")
    mock_speech.synthesize.return_value = b"ogg-audio"

    update = _make_update(user_id=2)
    context = _make_context({"language": "ta-IN"})

    await handle_voice(update, context)

    mock_speech.transcribe.assert_called_once_with(b"fake-wav", "ta-IN")


@pytest.mark.asyncio
async def test_handle_voice_no_voice_message():
    """If update has no voice, do nothing."""
    from bot import handle_voice

    update = _make_update(has_voice=False)
    context = _make_context()

    await handle_voice(update, context)

    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_voice_too_long():
    """Reject voice messages exceeding max duration."""
    from bot import handle_voice

    update = _make_update(voice_duration=999)
    context = _make_context({"language": "hi-IN"})

    await handle_voice(update, context)

    update.message.reply_text.assert_called_once()
    assert "too long" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
@patch("bot._ogg_to_wav", return_value=b"fake-wav")
@patch("bot.speech")
async def test_handle_voice_empty_transcript(mock_speech, mock_ogg):
    """When STT returns empty, inform the user."""
    from bot import handle_voice

    mock_speech.transcribe.return_value = ("", "hi-IN")

    update = _make_update()
    context = _make_context({"language": "hi-IN"})

    await handle_voice(update, context)

    # "Processing..." is the initial reply_text, then edit_text is called for the error
    assert update.message.reply_text.call_count == 1
    status_msg = update.message.reply_text.return_value
    status_msg.edit_text.assert_called_once()
    assert "couldn't understand" in status_msg.edit_text.call_args[0][0]


@pytest.mark.asyncio
@patch("bot._ogg_to_wav", return_value=b"fake-wav")
@patch("bot.speech")
@patch("bot.claude")
async def test_handle_voice_caption_truncated(mock_claude, mock_speech, mock_ogg):
    """Caption longer than 1024 chars gets truncated."""
    from bot import handle_voice

    mock_speech.transcribe.return_value = ("text", "es-ES")
    mock_claude.ask.return_value = ("full", "S" * 1100)
    mock_speech.synthesize.return_value = b"audio"

    update = _make_update()
    context = _make_context({"language": "es-ES"})

    await handle_voice(update, context)

    caption = update.message.reply_voice.call_args[1]["caption"]
    assert len(caption) <= 1024
    assert caption.endswith("...")


# ── language_callback ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_language_callback_sets_language():
    from bot import language_callback

    query = AsyncMock()
    query.data = "lang:hi-IN"
    update = AsyncMock()
    update.callback_query = query
    context = _make_context()

    await language_callback(update, context)

    assert context.user_data["language"] == "hi-IN"
    query.edit_message_text.assert_called_once()
    assert "Hindi" in query.edit_message_text.call_args[0][0]


@pytest.mark.asyncio
async def test_language_callback_auto_clears_language():
    from bot import language_callback

    query = AsyncMock()
    query.data = "lang:auto"
    update = AsyncMock()
    update.callback_query = query
    context = _make_context({"language": "ta-IN"})

    await language_callback(update, context)

    assert "language" not in context.user_data
    query.edit_message_text.assert_called_once()
    assert "auto-detect" in query.edit_message_text.call_args[0][0]


@pytest.mark.asyncio
async def test_language_callback_invalid_code_ignored():
    from bot import language_callback

    query = AsyncMock()
    query.data = "lang:xx-XX"
    update = AsyncMock()
    update.callback_query = query
    context = _make_context()

    await language_callback(update, context)

    assert "language" not in context.user_data
    query.edit_message_text.assert_not_called()


# ── clear ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("bot.claude")
async def test_clear_command(mock_claude):
    from bot import clear

    update = AsyncMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock()
    context = _make_context()

    await clear(update, context)

    mock_claude.clear_history.assert_called_once_with(42)
    update.message.reply_text.assert_called_once()
    assert "cleared" in update.message.reply_text.call_args[0][0]
