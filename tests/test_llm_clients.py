import pytest
import urllib.error

from local_agent.llm.base import LLMError, Message
from local_agent.llm.factory import create_llm_client
from local_agent.llm.fake import FakeLLMClient
from local_agent.llm.ollama import OllamaClient


pytestmark = pytest.mark.unit


def test_fake_llm_client_returns_canned_responses():
    client = FakeLLMClient(["first", "second"])

    assert client.complete([Message("user", "hello")]) == "first"
    assert client.complete([Message("user", "again")]) == "second"
    assert len(client.requests) == 2


def test_fake_llm_client_raises_when_exhausted():
    client = FakeLLMClient([])

    with pytest.raises(LLMError, match="no responses left"):
        client.complete([Message("user", "hello")])


def test_ollama_payload_formatting(monkeypatch):
    captured = {}
    client = OllamaClient(model="qwen", base_url="http://ollama.local")

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(client, "_post_json", fake_post)

    result = client.complete(
        [Message("system", "rules"), Message("user", "task")],
        temperature=0.2,
        max_tokens=123,
    )

    assert result == "ok"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "qwen"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0.2
    assert captured["payload"]["options"]["num_predict"] == 123
    assert captured["payload"]["messages"][1] == {"role": "user", "content": "task"}


def test_ollama_debug_payload_uses_streaming(monkeypatch):
    captured = {}
    client = OllamaClient(
        model="qwen",
        base_url="http://ollama.local",
        debug_stream=True,
    )

    def fake_post_stream(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr(client, "_post_stream", fake_post_stream)

    result = client.complete([Message("user", "task")])

    assert result == "ok"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["stream"] is True


def test_ollama_streaming_timeout_preserves_partial_response(monkeypatch, capsys):
    client = OllamaClient(model="qwen", debug_stream=True)

    class TimeoutStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield b'{"message": {"content": "partial"}, "done": false}\n'
            raise TimeoutError("slow model")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: TimeoutStream())

    with pytest.raises(LLMError, match="timed out") as excinfo:
        client.complete([Message("user", "task")])

    assert excinfo.value.partial_response == "partial"
    assert "partial" in capsys.readouterr().err


def test_ollama_malformed_response_raises(monkeypatch):
    client = OllamaClient(model="qwen")
    monkeypatch.setattr(client, "_post_json", lambda path, payload: {"message": {}})

    with pytest.raises(LLMError, match="Malformed Ollama response"):
        client.complete([Message("user", "task")])


def test_ollama_http_error_includes_status(monkeypatch):
    client = OllamaClient(model="qwen")

    def raise_http_error(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "not found",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)

    with pytest.raises(LLMError, match="HTTP 404"):
        client.complete([Message("user", "task")])


def test_create_ollama_client():
    client = create_llm_client("qwen", timeout_seconds=456.0)

    assert isinstance(client, OllamaClient)
    assert client.model == "qwen"
    assert client.timeout == 456.0


def test_create_ollama_client_debug_stream():
    client = create_llm_client("qwen", debug_stream=True)

    assert isinstance(client, OllamaClient)
    assert client.debug_stream is True


def test_create_fake_client(tmp_path):
    response_file = tmp_path / "response.py"
    response_file.write_text("print('ok')\n")

    client = create_llm_client(None, fake_response_file=response_file)

    assert isinstance(client, FakeLLMClient)


def test_create_ollama_client_requires_model():
    with pytest.raises(ValueError, match="--model is required"):
        create_llm_client(None)
