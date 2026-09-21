import json
import sys
import urllib.error
import urllib.request
from typing import Any

from local_agent.llm.base import LLMClient, LLMError, Message


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        debug_stream: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.debug_stream = debug_stream

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 12000,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": self.debug_stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.debug_stream:
            return self._post_stream("/api/chat", payload)
        response = self._post_json("/api/chat", payload)
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Malformed Ollama response: {response}") from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(
                f"Ollama request failed for {url}: HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Ollama request failed for {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned invalid JSON from {url}: {exc}") from exc

    def _post_stream(self, path: str, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        item = json.loads(raw_line.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        partial = "".join(chunks)
                        raise LLMError(
                            f"Ollama returned invalid streaming JSON from {url}: {exc}",
                            partial_response=partial,
                        ) from exc
                    content = ((item.get("message") or {}).get("content") or "")
                    if content:
                        chunks.append(content)
                        print(content, end="", file=sys.stderr, flush=True)
                    if item.get("done"):
                        print("", file=sys.stderr, flush=True)
                        return "".join(chunks)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(
                f"Ollama request failed for {url}: HTTP {exc.code}: {body}",
                partial_response="".join(chunks),
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Ollama request failed for {url}: {exc}",
                partial_response="".join(chunks),
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama request timed out for {url}: {exc}",
                partial_response="".join(chunks),
            ) from exc
        raise LLMError(
            f"Ollama stream ended without a done event from {url}",
            partial_response="".join(chunks),
        )
