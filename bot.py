import io
import logging
import tempfile

from pydub import AudioSegment
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN, SUPPORTED_LANGUAGES, MAX_AUDIO_DURATION_SECONDS
from services import speech, claude

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    lang_options = "\n".join(
        f"  • {cfg['label']} ({cfg['name']})"
        for cfg in SUPPORTED_LANGUAGES.values()
    )
    await update.message.reply_text(
        f"Welcome! Send me a voice message in one of these languages and I'll respond:\n\n"
        f"{lang_options}\n\n"
        f"Just record a voice message with your question and I'll reply with both voice and text!\n\n"
        f"Commands:\n"
        f"/start — Show this message\n"
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

    await update.message.reply_text("Processing your voice message...")

    try:
        # 1. Download the voice file (OGG format from Telegram)
        file = await voice.get_file()
        ogg_bytes = await file.download_as_bytearray()

        # 2. Convert OGG -> WAV for Google STT
        wav_bytes = _ogg_to_wav(bytes(ogg_bytes))

        # 3. Transcribe
        language_hint = context.user_data.get("language")
        transcript, detected_lang = speech.transcribe(wav_bytes, language_hint)

        if not transcript:
            await update.message.reply_text(
                "Sorry, I couldn't understand the audio. Please try again with a clearer recording."
            )
            return

        logger.info("Transcribed (%s): %s", detected_lang, transcript[:100])

        # Store detected language for future messages
        context.user_data["language"] = detected_lang

        # 4. Ask Claude (with conversation history)
        user_id = update.effective_user.id
        full_response, summary = claude.ask(transcript, detected_lang, user_id)

        # 5. Synthesize response to speech
        audio_bytes = speech.synthesize(full_response, detected_lang)

        # 6. Send voice response + text summary
        lang_name = SUPPORTED_LANGUAGES[detected_lang]["name"]
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
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
