"""Tests for services.claude — ask, clear_history, _parse_response."""

from unittest.mock import MagicMock, patch

import pytest


# ── _parse_response ─────────────────────────────────────────────────

def test_parse_response_with_both_sections():
    from services.claude import _parse_response

    text = "RESPONSE:\nThis is the answer.\n\nSUMMARY:\nShort summary."
    response, summary = _parse_response(text)
    assert response == "This is the answer."
    assert summary == "Short summary."


def test_parse_response_no_summary_section():
    from services.claude import _parse_response

    text = "RESPONSE:\nJust an answer with no summary marker."
    response, summary = _parse_response(text)
    assert response == "Just an answer with no summary marker."
    assert summary == "Just an answer with no summary marker."


def test_parse_response_no_markers():
    from services.claude import _parse_response

    text = "Plain text without any markers."
    response, summary = _parse_response(text)
    assert response == "Plain text without any markers."
    assert summary == "Plain text without any markers."


def test_parse_response_long_text_truncates_summary():
    from services.claude import _parse_response

    text = "A" * 300
    response, summary = _parse_response(text)
    assert response == text
    assert summary == "A" * 200 + "..."


def test_parse_response_summary_with_extra_whitespace():
    from services.claude import _parse_response

    text = "RESPONSE:\n  Spaced answer  \n\nSUMMARY:\n  Spaced summary  "
    response, summary = _parse_response(text)
    assert response == "Spaced answer"
    assert summary == "Spaced summary"


# ── helpers ─────────────────────────────────────────────────────────

def _mock_claude_response(text):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _get_messages_sent(mock_client):
    """Extract the messages list from the most recent API call."""
    return mock_client.messages.create.call_args.kwargs["messages"]


def _get_system_prompt(mock_client):
    """Extract the system prompt from the most recent API call."""
    return mock_client.messages.create.call_args.kwargs["system"]


# ── ask (with mocked Anthropic client) ──────────────────────────────

@patch("services.claude.client")
def test_ask_returns_response_and_summary(mock_client):
    import services.claude as mod
    mod.clear_history(1)

    mock_client.messages.create.return_value = _mock_claude_response(
        "RESPONSE:\nHola amigo.\n\nSUMMARY:\nSaludo."
    )

    response, summary = mod.ask("Hola", "es-ES", user_id=1)

    assert response == "Hola amigo."
    assert summary == "Saludo."
    assert "Spanish" in _get_system_prompt(mock_client)

    mod.clear_history(1)


@patch("services.claude.client")
def test_ask_sends_user_message(mock_client):
    import services.claude as mod
    mod.clear_history(1)

    # Capture messages at call time (before ask() appends assistant response)
    captured = {}
    def capture_call(**kwargs):
        captured["messages"] = [m.copy() for m in kwargs["messages"]]
        return _mock_claude_response("Reply")
    mock_client.messages.create.side_effect = capture_call

    mod.ask("Test query", "hi-IN", user_id=1)

    assert captured["messages"][-1] == {"role": "user", "content": "Test query"}

    mod.clear_history(1)


@patch("services.claude.client")
def test_ask_maintains_conversation_history(mock_client):
    import services.claude as mod
    mod.clear_history(2)

    mock_client.messages.create.return_value = _mock_claude_response("Reply 1")
    mod.ask("First question", "hi-IN", user_id=2)

    mock_client.messages.create.return_value = _mock_claude_response("Reply 2")
    mod.ask("Second question", "hi-IN", user_id=2)

    # On the second call, messages should contain: user1, assistant1, user2
    messages = _get_messages_sent(mock_client)
    assert messages[0] == {"role": "user", "content": "First question"}
    assert messages[1]["role"] == "assistant"
    assert messages[2] == {"role": "user", "content": "Second question"}

    mod.clear_history(2)


@patch("services.claude.CONVERSATION_HISTORY_LIMIT", 2)
@patch("services.claude.client")
def test_ask_trims_history_to_limit(mock_client):
    import services.claude as mod

    mod.clear_history(3)

    # Capture messages at each call time (before post-call mutation)
    call_snapshots = []
    def capture_call(**kwargs):
        call_snapshots.append([m.copy() for m in kwargs["messages"]])
        return _mock_claude_response("R")
    mock_client.messages.create.side_effect = capture_call

    # Send 3 exchanges with limit=2 — oldest should be trimmed
    mod.ask("Q1", "ta-IN", user_id=3)
    mod.ask("Q2", "ta-IN", user_id=3)
    mod.ask("Q3", "ta-IN", user_id=3)

    # On the third call, Q1 should have been trimmed
    third_call_messages = call_snapshots[2]
    user_messages = [m["content"] for m in third_call_messages if m["role"] == "user"]
    assert "Q1" not in user_messages
    assert "Q3" in user_messages
    assert third_call_messages[0]["role"] == "user"

    mod.clear_history(3)


# ── clear_history ───────────────────────────────────────────────────

@patch("services.claude.client")
def test_clear_history_resets_context(mock_client):
    import services.claude as mod

    mock_client.messages.create.return_value = _mock_claude_response("Reply")

    mod.ask("Hello", "es-ES", user_id=99)
    assert 99 in mod._conversation_history

    mod.clear_history(99)
    assert 99 not in mod._conversation_history


def test_clear_history_nonexistent_user_no_error():
    from services.claude import clear_history
    clear_history(999999)


# ── history isolation between users ─────────────────────────────────

@patch("services.claude.client")
def test_separate_history_per_user(mock_client):
    import services.claude as mod
    mod.clear_history(10)
    mod.clear_history(11)

    mock_client.messages.create.return_value = _mock_claude_response("Reply")

    mod.ask("User10 question", "ta-IN", user_id=10)
    mod.ask("User11 question", "es-ES", user_id=11)

    # User 11's call: only their own message should be sent as user content
    messages = _get_messages_sent(mock_client)
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "User11 question" in user_contents
    assert "User10 question" not in user_contents

    mod.clear_history(10)
    mod.clear_history(11)
