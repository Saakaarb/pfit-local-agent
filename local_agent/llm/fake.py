from collections import deque

from local_agent.llm.base import LLMClient, LLMError, Message


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[str]):
        self._responses = deque(responses)
        self.requests: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 12000,
    ) -> str:
        self.requests.append(messages)
        if not self._responses:
            raise LLMError("FakeLLMClient has no responses left")
        return self._responses.popleft()
