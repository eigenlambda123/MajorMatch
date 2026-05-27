import json

from api.orchestrator import run_orchestrated_assistant


def test_orchestrator_respects_allow_tool_calls_false():
    captured = {}

    def fake_chat_fn(messages, model=None, **kwargs):
        # record whether 'tools' was passed through
        captured['tools'] = kwargs.get('tools') if 'tools' in kwargs else None
        return {'message': {'content': 'Assistant reply without tools.'}}

    result = run_orchestrated_assistant(
        "I want recommendations.",
        model=None,
        allow_tool_calls=False,
        conversation_history=None,
        chat_fn=fake_chat_fn,
    )

    assert 'tools' in captured
    assert captured['tools'] is None
    assert result.reply


def test_orchestrator_respects_allow_tool_calls_true():
    captured = {}

    def fake_chat_fn(messages, model=None, **kwargs):
        captured['tools'] = kwargs.get('tools') if 'tools' in kwargs else None
        return {'message': {'content': 'Assistant reply with tools available.'}}

    result = run_orchestrated_assistant(
        "I want recommendations.",
        model=None,
        allow_tool_calls=True,
        conversation_history=None,
        chat_fn=fake_chat_fn,
    )

    assert 'tools' in captured
    assert isinstance(captured['tools'], list)
    assert result.reply
