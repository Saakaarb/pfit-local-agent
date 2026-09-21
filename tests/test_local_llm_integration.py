import os

import pytest

from local_agent.llm.base import Message
from local_agent.llm.factory import create_llm_client


pytestmark = pytest.mark.local_llm


def _local_llm_enabled():
    return os.environ.get("PFIT_RUN_LOCAL_LLM_TESTS") == "1"


@pytest.mark.skipif(
    not _local_llm_enabled(),
    reason="Set PFIT_RUN_LOCAL_LLM_TESTS=1 to run local LLM integration tests",
)
def test_configured_local_llm_returns_text():
    model = os.environ.get("PFIT_LLM_MODEL")
    base_url = os.environ.get("PFIT_LLM_BASE_URL")
    if model is None:
        pytest.skip("PFIT_LLM_MODEL must be set for local LLM integration tests")

    client = create_llm_client(model, base_url)
    response = client.complete(
        [
            Message(
                "user",
                "Reply with exactly this text and nothing else: pfit-local-ok",
            )
        ],
        temperature=0.0,
        max_tokens=16,
    )

    assert isinstance(response, str)
    assert response.strip()
