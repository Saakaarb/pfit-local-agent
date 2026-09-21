from pathlib import Path

from local_agent.llm.base import LLMClient
from local_agent.llm.fake import FakeLLMClient
from local_agent.llm.ollama import OllamaClient


def create_llm_client(
    model: str | None,
    base_url: str | None = None,
    fake_response_file: Path | None = None,
    timeout_seconds: float = 300.0,
    debug_stream: bool = False,
) -> LLMClient:
    if fake_response_file is not None:
        return FakeLLMClient([Path(fake_response_file).read_text()])

    if model is None:
        raise ValueError("--model is required for Ollama")

    return OllamaClient(
        model=model,
        base_url=base_url or "http://localhost:11434",
        timeout=timeout_seconds,
        debug_stream=debug_stream,
    )
