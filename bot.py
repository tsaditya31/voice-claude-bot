import asyncio
import io
import logging
import re
import tempfile

from pydub import AudioSegment
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import TELEGRAM_BOT_TOKEN, SUPPORTED_LANGUAGES, MAX_AUDIO_DURATION_SECONDS
from services import speech, claude

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _language_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"{cfg['label']} ({cfg['name']})", callback_data=f"lang:{code}")]
        for code, cfg in SUPPORTED_LANGUAGES.items()
    ]
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    lang_options = "\n".join(
        f"  • {cfg['label']} ({cfg['name']})"
        for cfg in SUPPORTED_LANGUAGES.values()
    )
    await update.message.reply_text(
        f"Welcome! I can understand voice messages in these languages:\n\n"
        f"{lang_options}\n\n"
        f"Just send a voice message — I'll auto-detect the language!\n"
        f"Or use /language to lock a specific language.\n\n"
        f"Commands:\n"
        f"/language — Set/change language (or use auto-detect)\n"
        f"/clear — Clear conversation history"
    )


async def language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /language command — change or clear language preference."""
    current = context.user_data.get("language")
    current_name = SUPPORTED_LANGUAGES.get(current, {}).get("name")
    status = f"Current: {current_name}" if current_name else "Current: Auto-detect"

    buttons = [
        [InlineKeyboardButton(f"{cfg['label']} ({cfg['name']})", callback_data=f"lang:{code}")]
        for code, cfg in SUPPORTED_LANGUAGES.items()
    ]
    buttons.append([InlineKeyboardButton("Auto-detect", callback_data="lang:auto")])

    await update.message.reply_text(
        f"{status}\n\nChoose a language or use auto-detect:",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from inline keyboard."""
    query = update.callback_query
    await query.answer()

    lang_code = query.data.removeprefix("lang:")

    if lang_code == "auto":
        context.user_data.pop("language", None)
        await query.edit_message_text(
            "Language set to auto-detect.\n\n"
            "Send a voice message and I'll figure out the language!\n\n"
            "Commands:\n"
            "/language — Change language\n"
            "/clear — Clear conversation history"
        )
        return

    if lang_code not in SUPPORTED_LANGUAGES:
        return

    context.user_data["language"] = lang_code
    lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]
    await query.edit_message_text(
        f"Language set to {lang_name}.\n\n"
        f"Send me a voice message and I'll reply with voice + text!\n\n"
        f"Commands:\n"
        f"/language — Change language\n"
        f"/clear — Clear conversation history"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command — reset conversation history."""
    user_id = update.effective_user.id
    claude.clear_history(user_id)
    await update.message.reply_text("Conversation history cleared.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming voice messages."""
    voice = update.message.voice
    if not voice:
        return

    if voice.duration > MAX_AUDIO_DURATION_SECONDS:
        await update.message.reply_text(
            f"Voice message too long (max {MAX_AUDIO_DURATION_SECONDS}s). Please send a shorter message."
        )
        return

    status_msg = await update.message.reply_text("Processing your voice message...")

    try:
        # 1. Download the voice file (OGG format from Telegram)
        file = await voice.get_file()
        ogg_bytes = await file.download_as_bytearray()

        # 2. Convert OGG -> WAV for Google STT
        wav_bytes = _ogg_to_wav(bytes(ogg_bytes))

        # 3. Transcribe — pass language hint if user set one, otherwise None for auto-detect
        language_hint = context.user_data.get("language")
        transcript, detected_lang = speech.transcribe(wav_bytes, language_hint)

        if not transcript:
            await status_msg.edit_text(
                "Sorry, I couldn't understand the audio. Please try again with a clearer recording."
            )
            return

        logger.info("Transcribed (%s): %s", detected_lang, transcript[:100])

        # 4. Start heartbeat — update status message every 2s while preparing response
        lang_name = SUPPORTED_LANGUAGES[detected_lang]["name"]
        heartbeat_task = asyncio.create_task(
            _heartbeat(status_msg, lang_name)
        )

        try:
            # 5. Ask Claude (with conversation history)
            user_id = update.effective_user.id
            full_response, summary = await asyncio.get_event_loop().run_in_executor(
                None, claude.ask, transcript, detected_lang, user_id
            )

            logger.info(
                "Claude response: %d chars, summary: %d chars",
                len(full_response), len(summary),
            )

            # 6. Synthesize response to speech
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, speech.synthesize, full_response, detected_lang
            )

            logger.info(
                "TTS audio: %d bytes (%.1f KB) for %d chars of text",
                len(audio_bytes), len(audio_bytes) / 1024, len(full_response),
            )
        finally:
            # Stop heartbeat
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # 7. Delete the status message
        try:
            await status_msg.delete()
        except Exception:
            pass

        # 8. Send voice response + text summary
        caption = f"[{lang_name}] {summary}"
        # Telegram caption limit is 1024 chars
        if len(caption) > 1024:
            caption = caption[:1021] + "..."

        await update.message.reply_voice(
            voice=io.BytesIO(audio_bytes),
            caption=caption,
        )

        # 9. Extract and send references (URLs, phone numbers, emails) as text
        references = _extract_references(full_response)
        if references:
            ref_text = "\n".join(references)
            await update.message.reply_text(
                f"📎 References:\n{ref_text}",
                disable_web_page_preview=True,
            )

    except Exception:
        logger.exception("Error processing voice message")
        try:
            await status_msg.edit_text(
                "Sorry, something went wrong processing your message. Please try again."
            )
        except Exception:
            await update.message.reply_text(
                "Sorry, something went wrong processing your message. Please try again."
            )


async def _heartbeat(status_msg, lang_name: str) -> None:
    """Edit the status message every 2s to show progress while preparing response."""
    dots = [".", "..", "..."]
    i = 0
    while True:
        await asyncio.sleep(2)
        try:
            await status_msg.edit_text(
                f"[{lang_name}] Preparing your response, please wait{dots[i % len(dots)]}"
            )
        except Exception:
            pass  # Message may have been deleted or edit conflicts — ignore
        i += 1


def _extract_references(text: str) -> list[str]:
    """Extract URLs, email addresses, and phone numbers from text."""
    refs = []

    # URLs (http/https/www)
    urls = re.findall(r'https?://[^\s<>\"\')\]]+|www\.[^\s<>\"\')\]]+', text)
    for url in urls:
        url = url.rstrip(".,;:!?")
        if url not in refs:
            refs.append(url)

    # Email addresses
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email in emails:
        if email not in refs:
            refs.append(email)

    # Phone numbers (international and local formats)
    phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text)
    for phone in phones:
        phone = phone.strip()
        # Only include if it looks like a real phone number (7+ digits)
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 7 and phone not in refs:
            refs.append(phone)

    return refs


def _ogg_to_wav(ogg_bytes: bytes) -> bytes:
    """Convert OGG audio to WAV (16kHz mono) for Google STT."""
    with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
        tmp.write(ogg_bytes)
        tmp.flush()
        audio = AudioSegment.from_ogg(tmp.name)

    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("language", language))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CallbackQueryHandler(language_callback, pattern=r"^lang:"))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
