from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from qwen_llm.config import ReasoningEffort


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class Engine(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def health(self) -> None: ...
    async def prepare(self, messages: list[dict], effort: ReasoningEffort, *, tools: list[dict] | None = None) -> list[int]: ...
    def generate(self, prompt: list[int], request_id: str, **sampling) -> AsyncIterator[Generation]: ...
    async def abort(self, request_id: str) -> None: ...
