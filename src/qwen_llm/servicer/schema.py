from typing import Literal
import json

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
from jsonschema import Draft202012Validator, SchemaError

from qwen_llm.config import ReasoningEffort


class Reasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effort: ReasoningEffort = Field(default=ReasoningEffort.NONE, description="原生档位；high 为 xhigh 的兼容别名")

    @field_validator("effort", mode="before",
                     json_schema_input_type=Literal["none", "low", "medium", "high", "xhigh"])
    @classmethod
    def canonical_effort(cls, value):
        return "xhigh" if value == "high" else value


class ChatTemplateKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_thinking: StrictBool = False
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE

    @field_validator("reasoning_effort", mode="before",
                     json_schema_input_type=Literal["none", "low", "medium", "high", "xhigh"])
    @classmethod
    def canonical_effort(cls, value):
        return "xhigh" if value == "high" else value


class PlainFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"] = "text"


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    json_schema: dict = Field(alias="schema")
    strict: StrictBool = False
    description: str | None = None

    @model_validator(mode="after")
    def valid_schema(self):
        try:
            Draft202012Validator.check_schema(self.json_schema)
        except SchemaError as exc:
            raise ValueError("无效的 JSON Schema") from exc
        return self


class TextOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: JsonSchemaFormat | PlainFormat = Field(default_factory=PlainFormat)


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
    seed: int | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None
    enable_thinking: StrictBool | None = None
    text: TextOptions = Field(default_factory=TextOptions)

    @model_validator(mode="after")
    def valid_input(self):
        if not self.input and not self.instructions:
            raise ValueError("input 与 instructions 至少提供一个")
        flags = [self.enable_thinking]
        efforts = [self.reasoning.effort] if "effort" in self.reasoning.model_fields_set else []
        if self.chat_template_kwargs is not None:
            kwargs = self.chat_template_kwargs
            if "enable_thinking" in kwargs.model_fields_set:
                flags.append(kwargs.enable_thinking)
            if "reasoning_effort" in kwargs.model_fields_set:
                efforts.append(kwargs.reasoning_effort)
        flags = [value for value in flags if value is not None]
        if len(set(efforts)) > 1:
            raise ValueError("reasoning.effort 与 chat_template_kwargs.reasoning_effort 冲突")
        effort = efforts[0] if efforts else ReasoningEffort.NONE
        if flags:
            if len(set(flags)) != 1:
                raise ValueError("enable_thinking 配置冲突")
            enabled = flags[0]
            if efforts and enabled != (effort is not ReasoningEffort.NONE):
                raise ValueError("reasoning 与 enable_thinking 冲突")
            if not efforts:
                effort = ReasoningEffort.XHIGH if enabled else ReasoningEffort.NONE
        self.reasoning = Reasoning(effort=effort)
        if self.text.format.type != "text" and self.reasoning.effort != "none":
            raise ValueError("结构化输出当前只支持非思考模式")
        self.messages()
        return self

    def messages(self) -> list[dict]:
        messages = []
        instructions = self.instructions or ""
        if self.text.format.type == "json_schema":
            # 采样语法约束不把字段说明交给模型。投影请求自带的同源 schema，
            # 让字段含义参与生成；业务指令与字段说明仍由调用方提供。
            instructions += '\n\nReturn JSON matching this schema and its field descriptions:\n' + json.dumps(
                self.text.format.json_schema, ensure_ascii=False, separators=(',', ':'))
        elif self.text.format.type == "json_object":
            instructions += '\n\nReturn a valid JSON object.'
        if instructions:
            messages.append({"role": "system", "content": instructions})
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
