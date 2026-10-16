from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Reasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effort: Literal["none", "low", "medium", "high"] = "none"


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input: str | list[dict] | None = None
    instructions: str | None = None
    stream: bool = False
    temperature: float = Field(default=0.6, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_output_tokens: int = Field(default=8192, gt=0, le=8192)
    reasoning: Reasoning = Field(default_factory=Reasoning)
    store: Literal[False] = False

    @model_validator(mode="after")
    def valid_input(self):
        if not self.input and not self.instructions:
            raise ValueError("input 与 instructions 至少提供一个")
        self.messages()
        return self

    def messages(self) -> list[dict]:
        messages = []
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        if isinstance(self.input, str):
            messages.append({"role": "user", "content": self.input})
        for item in self.input if isinstance(self.input, list) else []:
            if item.get("type", "message") != "message":
                raise ValueError("仅支持文本 message 历史，不支持工具或 reasoning 输入项")
            role = item.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise ValueError("不支持的 message role")
            content = item.get("content")
            if isinstance(content, list):
                if any(not isinstance(part, dict) or part.get("type") not in ("input_text", "output_text") or
                       not isinstance(part.get("text"), str) for part in content):
                    raise ValueError("仅支持文本 content parts")
                content = "".join(part["text"] for part in content)
            if not isinstance(content, str):
                raise ValueError("message content 必须为文本")
            messages.append({"role": "system" if role == "developer" else role, "content": content})
        return messages
