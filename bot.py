import io
import logging
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
    await update.message.reply_text(
        "Welcome! First, choose your language:",
        reply_markup=_language_keyboard(),
    )


async def language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /language command — change language."""
    current = context.user_data.get("language")
    current_name = SUPPORTED_LANGUAGES.get(current, {}).get("name", "not set")
    await update.message.reply_text(
        f"Current language: {current_name}\n\nChoose a new language:",
        reply_markup=_language_keyboard(),
    )


async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from inline keyboard."""
    query = update.callback_query
    await query.answer()

    lang_code = query.data.removeprefix("lang:")
    if lang_code not in SUPPORTED_LANGUAGES:
        return

    context.user_data["language"] = lang_code
    lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]
    await query.edit_message_text(
        f"Language set to {lang_name}.\n\n"
        f"Now send me a voice message and I'll reply with voice + text!\n\n"
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

    # Require language selection first
    user_lang = context.user_data.get("language")
    if not user_lang:
        await update.message.reply_text(
            "Please choose your language first:",
            reply_markup=_language_keyboard(),
        )
        return

    if voice.duration > MAX_AUDIO_DURATION_SECONDS:
        await update.message.reply_text(
            f"Voice message too long (max {MAX_AUDIO_DURATION_SECONDS}s). Please send a shorter message."
        )
        return

    await update.message.reply_text("Processing your voice message...")

    try:
        # 1. Download the voice file (OGG format from Telegram)
        file = await voice.get_file()
        ogg_bytes = await file.download_as_bytearray()

        # 2. Convert OGG -> WAV for Google STT
        wav_bytes = _ogg_to_wav(bytes(ogg_bytes))

        # 3. Transcribe using the user's chosen language
        transcript, detected_lang = speech.transcribe(wav_bytes, user_lang)

        if not transcript:
            await update.message.reply_text(
                "Sorry, I couldn't understand the audio. Please try again with a clearer recording."
            )
            return

        logger.info("Transcribed (%s): %s", detected_lang, transcript[:100])

        # 4. Ask Claude (with conversation history) — always use the user's chosen language
        user_id = update.effective_user.id
        full_response, summary = claude.ask(transcript, user_lang, user_id)

        # 5. Synthesize response to speech
        audio_bytes = speech.synthesize(full_response, user_lang)

        # 6. Send voice response + text summary
        lang_name = SUPPORTED_LANGUAGES[user_lang]["name"]
        caption = f"[{lang_name}] {summary}"
        # Telegram caption limit is 1024 chars
        if len(caption) > 1024:
            caption = caption[:1021] + "..."

        await update.message.reply_voice(
            voice=io.BytesIO(audio_bytes),
            caption=caption,
        )

    except Exception:
        logger.exception("Error processing voice message")
        await update.message.reply_text(
            "Sorry, something went wrong processing your message. Please try again."
        )


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
