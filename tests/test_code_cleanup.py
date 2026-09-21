import pytest

from local_agent.agent.code_cleanup import clean_llm_code_response


pytestmark = pytest.mark.unit


def test_clean_llm_code_response_strips_python_fence():
    response = "```python\nprint('ok')\n```\n"

    assert clean_llm_code_response(response) == "print('ok')"


def test_clean_llm_code_response_leaves_plain_code_unchanged():
    response = "print('ok')\n"

    assert clean_llm_code_response(response) == response


def test_clean_llm_code_response_does_not_strip_prose_wrapped_code():
    response = "Here is code:\n```python\nprint('ok')\n```"

    assert clean_llm_code_response(response) == response
