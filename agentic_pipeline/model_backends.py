"""Provider-neutral model calls for the agentic constraint pipelines.

The pipelines use one canonical conversation representation:

* user messages are ``{"role": "user", "content": "..."}``;
* assistant output items are :class:`ModelOutputItem` instances; and
* verifier results are OpenAI-shaped ``function_call_output`` dictionaries.

Each backend translates that representation to its provider's native API while
preserving raw assistant content. Preserving the raw content is important for
Anthropic models with adaptive thinking because signed thinking blocks must be
replayed unchanged when a tool-use conversation continues.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from anthropic import Anthropic, transform_schema
from openai import OpenAI

ProviderName = Literal["openai", "anthropic"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic")
DEFAULT_MAX_OUTPUT_TOKENS = 32_768


@dataclass(frozen=True)
class ModelOutputItem:
    """One provider response item with normalized tool-call attributes."""

    provider: ProviderName
    raw: Any
    type: str
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral response consumed by both constraint pipelines."""

    id: str | None
    output: list[ModelOutputItem]
    output_text: str
    usage: dict[str, Any] | None
    stop_reason: str | None = None


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal generation interface needed by the constraint agents."""

    provider: ProviderName

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Any],
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        output_name: str | None = None,
    ) -> ModelResponse:
        """Generate one assistant turn."""


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    dumped = _model_dump(usage)
    return dict(dumped) if not isinstance(dumped, dict) else dumped


class OpenAIBackend:
    """Native OpenAI Responses API adapter."""

    provider: ProviderName = "openai"

    def __init__(self, client: OpenAI | Any | None = None) -> None:
        self.client = client if client is not None else OpenAI()

    @staticmethod
    def _unwrap_input(items: Sequence[Any]) -> list[Any]:
        unwrapped: list[Any] = []
        for item in items:
            if isinstance(item, ModelOutputItem):
                if item.provider != "openai":
                    raise ValueError(
                        "cannot send Anthropic assistant state to OpenAI"
                    )
                unwrapped.append(item.raw)
            else:
                unwrapped.append(item)
        return unwrapped

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Any],
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        output_name: str | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": self._unwrap_input(input_items),
            "store": False,
        }
        if tools:
            kwargs.update(
                {
                    "tools": list(tools),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        if output_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_name or "structured_output",
                    "schema": output_schema,
                    "strict": True,
                }
            }

        response = self.client.responses.create(**kwargs)
        output: list[ModelOutputItem] = []
        for item in list(getattr(response, "output", []) or []):
            item_type = str(getattr(item, "type", "unknown"))
            output.append(
                ModelOutputItem(
                    provider=self.provider,
                    raw=item,
                    type=item_type,
                    name=getattr(item, "name", None),
                    arguments=getattr(item, "arguments", None),
                    call_id=getattr(item, "call_id", None),
                )
            )
        return ModelResponse(
            id=getattr(response, "id", None),
            output=output,
            output_text=getattr(response, "output_text", "") or "",
            usage=_usage_dict(getattr(response, "usage", None)),
            stop_reason=getattr(response, "status", None),
        )


class AnthropicBackend:
    """Native Anthropic Messages API adapter."""

    provider: ProviderName = "anthropic"

    def __init__(
        self,
        client: Anthropic | Any | None = None,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.client = client if client is not None else Anthropic()
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _flush_assistant(
        messages: list[dict[str, Any]],
        blocks: list[Any],
    ) -> None:
        if blocks:
            messages.append({"role": "assistant", "content": list(blocks)})
            blocks.clear()

    @staticmethod
    def _flush_tool_results(
        messages: list[dict[str, Any]],
        blocks: list[dict[str, Any]],
    ) -> None:
        if blocks:
            messages.append({"role": "user", "content": list(blocks)})
            blocks.clear()

    @classmethod
    def _messages(cls, items: Sequence[Any]) -> list[dict[str, Any]]:
        """Translate canonical items to Anthropic user/assistant messages."""
        messages: list[dict[str, Any]] = []
        assistant_blocks: list[Any] = []
        tool_results: list[dict[str, Any]] = []

        for item in items:
            if isinstance(item, ModelOutputItem):
                if item.provider != "anthropic":
                    raise ValueError(
                        "cannot send OpenAI assistant state to Anthropic"
                    )
                cls._flush_tool_results(messages, tool_results)
                assistant_blocks.append(_model_dump(item.raw))
                continue

            cls._flush_assistant(messages, assistant_blocks)
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
            ):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": item["call_id"],
                        "content": str(item.get("output", "")),
                    }
                )
                continue

            cls._flush_tool_results(messages, tool_results)
            if not isinstance(item, dict) or item.get("role") not in {
                "user",
                "assistant",
            }:
                raise ValueError(f"unsupported canonical input item: {item!r}")
            messages.append(
                {
                    "role": item["role"],
                    "content": item.get("content", ""),
                }
            )

        cls._flush_assistant(messages, assistant_blocks)
        cls._flush_tool_results(messages, tool_results)
        return messages

    @staticmethod
    def _tools(
        tools: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        translated: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                raise ValueError(
                    f"Anthropic adapter only supports function tools: {tool!r}"
                )
            parameters = (
                transform_schema(tool["parameters"])
                if tool.get("strict")
                else tool["parameters"]
            )
            translated_tool: dict[str, Any] = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": parameters,
            }
            if tool.get("strict"):
                translated_tool["strict"] = True
            translated.append(translated_tool)
        return translated

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Any],
        tools: Sequence[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        output_name: str | None = None,
    ) -> ModelResponse:
        del output_name  # Anthropic structured outputs do not accept a name.
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "system": instructions,
            "messages": self._messages(input_items),
        }
        if tools:
            kwargs["tools"] = self._tools(tools)
            kwargs["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }
        if output_schema is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": transform_schema(output_schema),
                }
            }

        # Anthropic requires streaming when a request's configured output
        # budget could take more than ten minutes. The pipeline still consumes
        # one complete response after the SDK has assembled the stream.
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        output: list[ModelOutputItem] = []
        text_parts: list[str] = []
        for block in list(getattr(response, "content", []) or []):
            block_type = str(getattr(block, "type", "unknown"))
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            if block_type == "tool_use":
                name = getattr(block, "name", None)
                call_id = getattr(block, "id", None)
                arguments = json.dumps(
                    getattr(block, "input", {}),
                    ensure_ascii=False,
                )
                normalized_type = "function_call"
            else:
                name = None
                call_id = None
                arguments = None
                normalized_type = block_type
            output.append(
                ModelOutputItem(
                    provider=self.provider,
                    raw=block,
                    type=normalized_type,
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                )
            )
        return ModelResponse(
            id=getattr(response, "id", None),
            output=output,
            output_text="".join(text_parts),
            usage=_usage_dict(getattr(response, "usage", None)),
            stop_reason=getattr(response, "stop_reason", None),
        )


def create_backend(
    provider: ProviderName,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> ModelBackend:
    """Construct a native provider backend after validating credentials."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIBackend()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicBackend(max_output_tokens=max_output_tokens)
    raise ValueError(
        f"unsupported provider {provider!r}; choose from {SUPPORTED_PROVIDERS}"
    )


def ensure_backend(client_or_backend: Any) -> ModelBackend:
    """Accept the new backend protocol or wrap a legacy OpenAI test client."""
    if isinstance(client_or_backend, ModelBackend):
        return client_or_backend
    if hasattr(client_or_backend, "responses"):
        return OpenAIBackend(client_or_backend)
    raise TypeError("expected a ModelBackend or OpenAI-compatible client")
