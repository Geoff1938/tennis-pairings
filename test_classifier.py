"""Tests for the Haiku-based command classifier in admin_bot.

The classifier itself is one short Haiku call — drive it with a
stub client so the tests don't touch the real API."""

from __future__ import annotations

from types import SimpleNamespace

from admin_bot import (
    CLASSIFIER_SYSTEM_PROMPT,
    MODEL_HAIKU,
    MODEL_SONNET,
    _classify_command,
)


class _FakeContent:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class _FakeResponse:
    def __init__(self, text: str, *, input_tokens: int = 200,
                 output_tokens: int = 2):
        self.content = [_FakeContent(text)]
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )


class _FakeClient:
    """Drop-in for anthropic.Anthropic with just .messages.create."""
    def __init__(self, response_text: str = "SIMPLE", *, raise_exc=None):
        self._response_text = response_text
        self._raise = raise_exc
        self.calls: list[dict] = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._response_text)


# ---------- happy path -----------------------------------------------------


def test_simple_command_routes_to_haiku():
    client = _FakeClient("SIMPLE")
    model, usage = _classify_command(
        client, "boris add Joe Bloggs", has_history=False,
    )
    assert model == MODEL_HAIKU
    # Classifier did call the API.
    assert len(client.calls) == 1
    # Usage is populated from the response.
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 2


def test_complex_command_routes_to_sonnet():
    client = _FakeClient("COMPLEX")
    model, _ = _classify_command(
        client, "boris kickoff", has_history=False,
    )
    assert model == MODEL_SONNET


# ---------- conservative defaults ------------------------------------------


def test_history_present_skips_classifier_and_uses_sonnet():
    """A short follow-up like 'court 5' needs context the classifier
    can't see — always route to Sonnet when history is present."""
    client = _FakeClient("SIMPLE")  # would say simple if asked
    model, usage = _classify_command(
        client, "court 5", has_history=True,
    )
    assert model == MODEL_SONNET
    # Critically: no API call made (saves money + latency).
    assert client.calls == []
    # Empty usage when classifier is skipped.
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_classifier_failure_falls_back_to_sonnet():
    """Any error from the classifier → use Sonnet. Safer to over-spend
    on one command than route to the cheaper model and get wrong
    behaviour."""
    client = _FakeClient(raise_exc=RuntimeError("api down"))
    model, _ = _classify_command(
        client, "boris add Joe", has_history=False,
    )
    assert model == MODEL_SONNET


def test_malformed_classifier_response_falls_back_to_sonnet():
    """If Haiku returns something other than SIMPLE/COMPLEX (e.g.
    rambles an answer), default to Sonnet rather than guess."""
    client = _FakeClient("I think this is probably SIMPLE...")
    # Note: starts with 'I', not 'SIMPLE' — should NOT route to Haiku.
    model, _ = _classify_command(
        client, "boris what's tonight", has_history=False,
    )
    assert model == MODEL_SONNET


def test_simple_with_trailing_whitespace_is_recognised():
    client = _FakeClient("SIMPLE\n")
    model, _ = _classify_command(
        client, "boris help", has_history=False,
    )
    assert model == MODEL_HAIKU


def test_lowercase_simple_is_recognised():
    """The prompt asks for uppercase but be tolerant."""
    client = _FakeClient("simple")
    model, _ = _classify_command(
        client, "boris add Joe", has_history=False,
    )
    assert model == MODEL_HAIKU


# ---------- classifier API call shape --------------------------------------


def test_classifier_uses_haiku_model_and_caches_prompt():
    client = _FakeClient("SIMPLE")
    _classify_command(client, "boris add Joe", has_history=False)
    call = client.calls[0]
    assert call["model"] == MODEL_HAIKU
    # Max output cap is tiny — only need one word back.
    assert call["max_tokens"] <= 16
    # System is a list with cache_control set so the classifier prompt
    # is cached and basically free after the first call.
    system = call["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == CLASSIFIER_SYSTEM_PROMPT
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # The command text is the user message.
    assert call["messages"] == [
        {"role": "user", "content": "boris add Joe"},
    ]
