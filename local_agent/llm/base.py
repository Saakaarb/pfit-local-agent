from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMError(RuntimeError):
    """Raised when a local Ollama request fails."""

    def __init__(self, message: str, partial_response: str = ""):
        super().__init__(message)
        self.partial_response = partial_response


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 12000,
    ) -> str:
        """Return a completion for chat-style messages."""
