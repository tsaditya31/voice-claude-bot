import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CONVERSATION_HISTORY_LIMIT
from services.language import get_language_name

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-user conversation history: {user_id: [{"role": ..., "content": ...}, ...]}
_conversation_history: dict[int, list[dict]] = {}

# Track last detected language per user to reset history on language switch
_user_language: dict[int, str] = {}

SYSTEM_PROMPT = """You are a helpful multilingual assistant. The user will send you a query in {language_name} ({language_code}).

Instructions:
1. Understand the query (it will be in {language_name}).
2. Provide a helpful, accurate answer.
3. Write your ENTIRE response in {language_name} — do NOT use English unless the user explicitly asks for it.
4. Structure your response as:
   RESPONSE:
   <your full detailed answer in {language_name}>

   SUMMARY:
   <a brief 2-3 sentence summary of your answer in {language_name}>"""


def ask(query_text: str, language_code: str, user_id: int) -> tuple[str, str]:
    """Send a query to Claude and get a response in the same language.

    Maintains conversation history per user (last CONVERSATION_HISTORY_LIMIT exchanges).
    Returns (full_response_text, summary_text) both in the original language.
    """
    language_name = get_language_name(language_code)

    system = SYSTEM_PROMPT.format(
        language_name=language_name,
        language_code=language_code,
    )

    # Clear history when user switches language to avoid cross-language confusion
    prev_lang = _user_language.get(user_id)
    if prev_lang and prev_lang != language_code:
        _conversation_history.pop(user_id, None)
    _user_language[user_id] = language_code

    # Get or create conversation history for this user
    history = _conversation_history.setdefault(user_id, [])

    # Add the new user message
    history.append({"role": "user", "content": query_text})

    # Trim to last N exchanges (each exchange = 1 user + 1 assistant = 2 messages)
    max_messages = CONVERSATION_HISTORY_LIMIT * 2
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    # Ensure history starts with a user message (required by the API)
    while history and history[0]["role"] != "user":
        history.pop(0)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system,
        messages=history,
    )

    full_text = response.content[0].text

    # Append assistant response to history
    history.append({"role": "assistant", "content": full_text})

    # Trim again after adding the response
    if len(history) > max_messages:
        history[:] = history[-max_messages:]
    while history and history[0]["role"] != "user":
        history.pop(0)

    full_response, summary = _parse_response(full_text)
    return full_response, summary


def clear_history(user_id: int) -> None:
    """Clear conversation history for a user."""
    _conversation_history.pop(user_id, None)
    _user_language.pop(user_id, None)


def _parse_response(text: str) -> tuple[str, str]:
    """Parse the RESPONSE: and SUMMARY: sections from Claude's output."""
    response_part = text
    summary_part = ""

    if "SUMMARY:" in text:
        parts = text.split("SUMMARY:", 1)
        response_part = parts[0].strip()
        summary_part = parts[1].strip()

    if response_part.startswith("RESPONSE:"):
        response_part = response_part[len("RESPONSE:"):].strip()

    if not summary_part:
        summary_part = response_part[:200]
        if len(response_part) > 200:
            summary_part += "..."

    return response_part, summary_part
