"""
LLM API wrapper — the only module that knows about providers.

Everything else in StudyMate calls LLMClient.chat() with an OpenAI-style
message list and receives a _MessageShim back.  Swapping providers means
changing only env vars, not application code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised for all user-visible API failures."""


# ---------------------------------------------------------------------------
# Unified response shim
# ---------------------------------------------------------------------------
# Both SDK response objects are structurally incompatible, so we normalise
# every response into these three dataclasses before returning to callers.

@dataclass
class _FunctionShim:
    name: str
    arguments: str  # JSON-encoded string, matching the OpenAI convention


@dataclass
class _ToolCallShim:
    id: str
    function: _FunctionShim
    type: str = "function"


@dataclass
class _MessageShim:
    content: str | None
    tool_calls: list[_ToolCallShim] | None = field(default=None)


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

def _translate_anthropic_error(exc: anthropic.APIError) -> LLMError:
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMError("Anthropic authentication failed — check STUDYMATE_API_KEY.")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMError("Anthropic rate limit hit — wait and retry, or switch to a smaller model.")
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError("Anthropic API took too long — try again.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError("Could not reach Anthropic — check your internet connection.")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMError("Anthropic model not found — check STUDYMATE_MODEL.")
    if isinstance(exc, anthropic.BadRequestError):
        return LLMError(f"Anthropic rejected the request: {exc.message}")
    return LLMError(f"Anthropic API error: {exc}")


def _translate_openai_error(exc: openai.APIError) -> LLMError:
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("Authentication failed — check STUDYMATE_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        return LLMError("Rate limit hit — wait and retry, or switch to a smaller model.")
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("API took too long — try again.")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("Could not reach the API — check your internet connection.")
    if isinstance(exc, openai.NotFoundError):
        return LLMError("Model not found — check STUDYMATE_MODEL.")
    if isinstance(exc, openai.BadRequestError):
        return LLMError(f"API rejected the request: {exc.message}")
    return LLMError(f"API error: {exc}")


# ---------------------------------------------------------------------------
# Anthropic message conversion
# ---------------------------------------------------------------------------
# The Anthropic SDK uses a different message schema than OpenAI.  We store
# history in OpenAI format (simpler to reason about) and convert on the fly.

def _to_anthropic_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-style messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_prompt: str | None = None
    converted: list[dict] = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        # Step 1 — pull system messages out entirely.
        # Anthropic takes "system" as a top-level parameter, not a message.
        if role == "system":
            system_prompt = msg["content"]
            i += 1
            continue

        # Step 2 — convert assistant messages that contain tool calls.
        # OpenAI encodes tool calls in a "tool_calls" list on the message.
        # Anthropic expects a content block array with type="tool_use".
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            content_blocks: list[dict] = []

            text = msg.get("content") or ""
            # Anthropic requires assistant content to be non-empty when tool_use
            # blocks are also present.  Use a single space as a safe placeholder.
            if text or not tool_calls:
                content_blocks.append({"type": "text", "text": text or " "})

            for tc in tool_calls:
                try:
                    tool_input = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                if not isinstance(tool_input, dict):
                    tool_input = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": tool_input,
                })

            converted.append({"role": "assistant", "content": content_blocks})
            i += 1
            continue

        # Step 3 — batch consecutive tool result messages into a single user turn.
        # Anthropic requires tool results to live inside a user message as
        # "tool_result" content blocks.  OpenAI sends them as separate messages
        # with role="tool".  We consume the entire run of tool messages at once
        # so there is never more than one user message per assistant turn —
        # Anthropic rejects conversations where two consecutive messages share
        # the same role.
        if role == "tool":
            tool_result_blocks: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                t = messages[i]
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content": t["content"],
                })
                i += 1
            converted.append({"role": "user", "content": tool_result_blocks})
            continue

        # Plain user message — content is always a string here.
        converted.append({"role": "user", "content": msg["content"]})
        i += 1

    return system_prompt, converted


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------

def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI function-calling schema to Anthropic tool schema."""
    result = []
    for t in tools:
        fn = t["function"]
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "anthropic": "claude-opus-4-7",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o",
}

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient:
    """Provider-agnostic LLM client.

    Reads configuration from environment variables unless a config dict is
    passed explicitly (useful for testing or multi-client setups).

    config keys: api_key, provider, model, base_url
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        cfg = config or {}

        api_key = cfg.get("api_key") or os.getenv("STUDYMATE_API_KEY")
        if not api_key:
            raise LLMError(
                "STUDYMATE_API_KEY is not set. "
                "Copy .env.example to .env and add your API key."
            )

        self.provider = (
            cfg.get("provider") or os.getenv("STUDYMATE_PROVIDER") or "anthropic"
        ).lower()

        if self.provider not in _DEFAULTS:
            raise LLMError("STUDYMATE_PROVIDER must be anthropic, groq, or openai.")

        default_model = _DEFAULTS.get(self.provider, _DEFAULTS["anthropic"])
        self.model = cfg.get("model") or os.getenv("STUDYMATE_MODEL") or default_model

        # Auto-populate Groq's base URL so callers don't have to remember it.
        base_url = cfg.get("base_url") or os.getenv("STUDYMATE_BASE_URL") or ""
        if self.provider == "groq" and not base_url:
            base_url = _GROQ_BASE_URL

        if self.provider == "anthropic":
            self._anthropic = anthropic.Anthropic(api_key=api_key)
        else:
            # Both Groq and OpenAI use the openai SDK; only base_url differs.
            self._openai = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or None,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> _MessageShim:
        """Send a message list and return a normalised _MessageShim.

        Args:
            messages:    OpenAI-style message list (role/content/tool_calls).
            tools:       OpenAI function-calling schema list, or None.
            temperature: Sampling temperature (ignored for Anthropic — see below).
            max_tokens:  Maximum tokens in the completion.
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, max_tokens)
        return self._chat_openai(messages, tools, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
    ) -> _MessageShim:
        system_prompt, anthropic_messages = _to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        # Temperature is intentionally omitted.
        # Leave sampling to provider defaults; some models reject this parameter.

        try:
            response = self._anthropic.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise _translate_anthropic_error(exc) from exc

        text_blocks: list[str] = []
        tool_calls: list[_ToolCallShim] = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    _ToolCallShim(
                        id=block.id,
                        function=_FunctionShim(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        return _MessageShim(
            content="\n".join(text_blocks) or None,
            tool_calls=tool_calls or None,
        )

    def _chat_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> _MessageShim:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._openai.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise _translate_openai_error(exc) from exc

        msg = response.choices[0].message
        raw_tool_calls = msg.tool_calls or []

        tool_calls: list[_ToolCallShim] = [
            _ToolCallShim(
                id=tc.id,
                function=_FunctionShim(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in raw_tool_calls
        ]

        return _MessageShim(
            content=msg.content,
            tool_calls=tool_calls or None,
        )
