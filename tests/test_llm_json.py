import pytest

from local_agent.agent.llm_json import parse_llm_json_object


pytestmark = pytest.mark.unit


def test_parse_llm_json_object_extracts_prose_wrapped_json():
    data = parse_llm_json_object(
        'Here is the result:\n{"ok": true, "items": [1]}\nDone.',
        "test-step",
    )

    assert data == {"ok": True, "items": [1]}


def test_parse_llm_json_object_accepts_fenced_json():
    data = parse_llm_json_object(
        '```json\n{"critical_errors": []}\n```',
        "test-step",
    )

    assert data == {"critical_errors": []}
