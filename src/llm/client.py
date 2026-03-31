"""Abstracted LLM client for model-agnostic AI interactions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import anthropic


class Role(Enum):
    """LLM roles with different information visibility."""

    REFEREE = "referee"  # Full knowledge: all hidden info, full state, full rules
    RULES_INTERPRETER = "rules_interpreter"  # AI hidden + visible state + rules
    STRATEGIST = "strategist"  # AI hidden + visible state + legal moves + strategy


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    raw_response: Any
    model: str
    input_tokens: int
    output_tokens: int

    def as_json(self) -> dict:
        """Parse content as JSON, stripping markdown code fences if present."""
        import json

        text = self.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines[1:] if not l.strip() == "```"]
            text = "\n".join(lines)
        return json.loads(text)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: The user's message/query.
            role: Optional role hint for logging/debugging.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with the model's output.
        """
        pass

    @abstractmethod
    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: The user's message/query.
            images: List of image bytes (PNG, JPEG, etc.).
            role: Optional role hint for logging/debugging.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with the model's output.
        """
        pass


class ClaudeClient(LLMClient):
    """Claude API client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model identifier to use.
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and get a completion."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def complete_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes],
        *,
        role: Role | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt with images and get a completion."""
        import base64

        # Build content with images and text
        content: list[dict] = []

        for image_bytes in images:
            # Detect media type from bytes
            media_type = self._detect_media_type(image_bytes)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )

        content.append({"type": "text", "text": user_prompt})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @staticmethod
    def _detect_media_type(image_bytes: bytes) -> str:
        """Detect image media type from magic bytes."""
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif image_bytes[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif image_bytes[:4] == b"GIF8":
            return "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        else:
            # Default to JPEG if unknown
            return "image/jpeg"


def create_client(
    provider: str = "claude",
    api_key: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: The LLM provider ("claude" for now, extensible later).
        api_key: API key for the provider.
        model: Model identifier to use.

    Returns:
        An LLMClient instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider == "claude":
        return ClaudeClient(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
